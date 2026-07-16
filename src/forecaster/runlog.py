"""Freeze an agent RunResult as durable provenance (M4 persistence layer).

Two halves, split per the DB rule:
  - QUERYABLE (DuckDB, via store.py): a `runs` row + the emitted TAF archived in `tafs`
    + the worksheet in `taf_worksheets`.
  - BLOB (a file): the full frozen `messages` array -- prompt + reasoning + every tool
    call and the weather data it returned + the base64 images the model saw. Big and
    self-contained, so it lives on disk, referenced by runs.transcript_path.

persist_run ties them together and writes a PENDING evaluation: scores come LATER
(score_taf.py --pending), once the validity window elapses and obs accumulate. At
collection time the truth does not exist yet -- that ordering is what makes the
benchmark leakage-proof. No LLM, no network; testable with a synthetic RunResult.
"""

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from forecaster import store, tafgen
from forecaster.agent import RunResult
from forecaster.config import settings
from forecaster.tafarchive import build_taf_row


def _artifacts_root(db_path: str | None = None) -> Path:
    """Transcript blobs live next to the DB, under data/runs/ (data/ is gitignored)."""
    return Path(db_path or settings.db_path).parent / "runs"


def toolset_hash(names: list[str]) -> str:
    """Stable sha12 over the SORTED tool names offered in a run (order-independent), so a
    run's toolset is a comparable fingerprint on the run row."""
    return hashlib.sha256(",".join(sorted(names)).encode()).hexdigest()[:12]


def config_id_for(*, model: str, temperature: float | None, max_tokens: int | None,
                  seed: int | None, worksheet_mode: str, evidence_mode: str | None,
                  toolset_hash: str) -> str:
    """Hash the full benchmark matrix cell into a stable config id. Two runs collide on
    config_id ONLY if every knob matches -- so temp-0 vs temp-0.2, or a different toolset,
    are distinct cells. The old f'{model}:{worksheet_mode}' scheme silently merged them."""
    payload = json.dumps({
        "model": model, "temperature": temperature, "max_tokens": max_tokens,
        "seed": seed, "worksheet_mode": worksheet_mode, "evidence_mode": evidence_mode,
        "toolset_hash": toolset_hash,
    }, sort_keys=True)
    return "cfg_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def write_transcript(run_id: str, res: RunResult, *, artifacts_dir: str | None = None,
                     db_path: str | None = None) -> Path:
    """Dump the full frozen messages array to <root>/<run_id>/messages.json.gz and return
    the path. Images already ride inside messages as base64 data URLs, so the file is a
    self-contained, replayable snapshot of exactly what the model saw -- gzipped because
    those base64 blobs dominate the size and compress well."""
    root = Path(artifacts_dir) if artifacts_dir else _artifacts_root(db_path)
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / "messages.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(res.messages, f, indent=2)
    return path


def read_transcript(path: str | Path) -> list[dict]:
    """Load a transcript blob back into the messages array, transparently handling a
    gzipped (.gz) or a plain .json file -- so a replay reader need not care which it is."""
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(p.read_text(encoding="utf-8"))


def persist_run(
    res: RunResult,
    *,
    run_id: str,
    station: str,
    issue_time: datetime,
    valid_from: datetime,
    valid_to: datetime,
    worksheet_mode: str,
    config_id: str | None = None,
    experiment_id: str | None = None,
    harness_git_sha: str | None = None,
    model: str | None = None,
    evidence_mode: str | None = None,
    db_path: str | None = None,
    artifacts_dir: str | None = None,
) -> dict:
    """Persist one agent run: transcript file + `runs` row + archived TAF + worksheet +
    a PENDING evaluation. Idempotent by run_id (re-persisting replaces, never duplicates).
    Returns a summary of the ids written. `issue_time` anchors the emitted TAF's calendar."""
    model = model or res.model
    th = toolset_hash(res.toolset_names)
    config_id = config_id or config_id_for(
        model=model, temperature=res.temperature, max_tokens=res.max_tokens,
        seed=res.seed, worksheet_mode=worksheet_mode, evidence_mode=evidence_mode,
        toolset_hash=th)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    transcript_path = write_transcript(run_id, res, artifacts_dir=artifacts_dir, db_path=db_path)

    con = store.connect(db_path) if db_path else store.connect()
    try:
        store.init_scoring_schema(con)
        store.init_worksheet_schema(con)
        store.init_runs_schema(con)

        product = res.final_taf or res.last_taf     # prefer the clean emit; fall back to last

        # Worksheet first, so the archived TAF can carry its worksheet_id.
        worksheet_id = None
        if res.worksheet is not None:
            worksheet_id = f"{run_id}_ws"
            ws = res.worksheet
            store.insert_worksheet(
                con, worksheet_id=worksheet_id, worksheet_json=ws.model_dump_json(),
                station=station,
                forecast_type=(ws.task.forecast_type if ws.task else None),
                valid_from_utc=(ws.task.valid_from_utc if ws.task else None),
                valid_to_utc=(ws.task.valid_to_utc if ws.task else None),
                mode=worksheet_mode, evidence_mode=evidence_mode, model=model,
                final_taf_text=(tafgen.render_taf(product) if product is not None else None),
                taf_product_json=(product.model_dump_json() if product is not None else None),
                checker_findings_json=json.dumps(res.worksheet_findings),
                status="accepted", evidence=res.evidence)

        # Archive the emitted TAF. A malformed last_taf that will not render/parse is
        # still RECORDED on the run (taf_id NULL) -- it just cannot be scored.
        taf_id = None
        if product is not None:
            try:
                raw = tafgen.render_taf(product)
                row = build_taf_row(raw, issue_ref=issue_time, producer_kind="artificial",
                                    producer_name=model, source="agent_run", canonical=True)
                row["run_id"] = run_id
                row["experiment_id"] = experiment_id
                row["worksheet_id"] = worksheet_id
                row["taf_product_json"] = product.model_dump_json()
                store.insert_taf(con, row)
                taf_id = row["taf_id"]
            except Exception:  # noqa: BLE001 -- an unrenderable/unparseable TAF is not a crash
                taf_id = None

        # Pending evaluation: only meaningful when there is a TAF to score.
        evaluation_id = None
        if taf_id is not None:
            evaluation_id = run_id
            store.insert_evaluation(con, {
                "evaluation_id": evaluation_id, "station": station,
                "valid_from": valid_from, "valid_to": valid_to,
                "status": "pending", "created_at": now})

        store.insert_run(con, {
            "run_id": run_id, "experiment_id": experiment_id, "station": station,
            "issue_time_utc": issue_time, "valid_from_utc": valid_from, "valid_to_utc": valid_to,
            "producer_kind": "artificial", "model": model,
            "served_model": " | ".join(res.served_models) or None,
            "system_fingerprint": " | ".join(res.system_fingerprints) or None,
            "base_url": res.base_url, "temperature": res.temperature,
            "max_tokens": res.max_tokens, "seed": res.seed, "toolset_hash": th,
            "worksheet_mode": worksheet_mode,
            "config_id": config_id, "harness_git_sha": harness_git_sha,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "n_steps": len(res.steps), "n_tool_calls": sum(res.used.values()),
            "tools_used_json": json.dumps(dict(res.used)), "stop_reason": res.stop_reason,
            "convergence": res.convergence, "first_emit_step": res.first_emit_step,
            "nudge_step": res.nudge_step, "taf_id": taf_id,
            "taf_clean": res.final_taf is not None, "worksheet_id": worksheet_id,
            "transcript_path": str(transcript_path), "fatal": res.fatal, "created_at": now})
    finally:
        con.close()

    return {"run_id": run_id, "taf_id": taf_id, "worksheet_id": worksheet_id,
            "evaluation_id": evaluation_id, "transcript_path": str(transcript_path)}
