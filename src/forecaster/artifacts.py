"""Content-addressed blob store for frozen agent inputs -- the STORAGE half of the
artifact store (docs/artifact_store.md).

Split deliberately from the INDEX half, which is three tables in store.py: store.py is the
only file allowed to touch DuckDB, and the standing rule is that images never enter the
relational DB. So bytes live here, records and references live there, and neither module
has to know how the other works.

    blobs/<sha256[0:2]>/<sha256>.<ext>

A blob is written ONCE and never mutated or renamed (contract rule 5). Re-capturing
identical bytes is a no-op on disk plus one new row in `artifact_keys`, and that is the
whole economy of the archive: 22 stations issue at 11Z and share one CONUS water-vapour
image, 16 satellite regions cover 71 stations, and the same GFS panel serves every station
on its domain. Keying on station instead would store the same picture 48 times.

The extension is cosmetic -- the sha is the address. It exists so a human sorting through
`data/archive/blobs` can open a file, and so a replay can hand the right mime to a model.
"""

import hashlib
from pathlib import Path

from forecaster.config import settings

# (leading bytes, mime, extension). Order matters only in that the first match wins; these
# signatures do not overlap. We sniff rather than trust the provider's Content-Type because
# several of ours lie: IEM's national radar silently degrades from PNG to the NWS RIDGE GIF,
# and SLIDER serves PNG under a .jpg-looking path.
_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png", "png"),
    (b"\xff\xd8\xff", "image/jpeg", "jpg"),
    (b"GIF87a", "image/gif", "gif"),
    (b"GIF89a", "image/gif", "gif"),
)


def sniff(data: bytes) -> tuple[str, str]:
    """(mime, ext) from the bytes themselves. Anything unrecognised is treated as text,
    which is correct for the two text artifacts we hold (a TAF bulletin and a raw BUFR
    ascent) and harmless for anything else -- the sha still addresses it exactly."""
    for magic, mime, ext in _SIGNATURES:
        if data.startswith(magic):
            return mime, ext
    # mp4 puts a box length in the first 4 bytes and the brand at offset 4, so it cannot be
    # matched by a prefix. Loops archive FRAMES (rule 3), so this only fires if someone
    # stores a composed video deliberately.
    if len(data) > 11 and data[4:8] == b"ftyp":
        return "video/mp4", "mp4"
    return "text/plain", "txt"


def archive_root(root: str | Path | None = None) -> Path:
    """The archive directory. Defaults beside the DuckDB file, so `data/` stays the one
    gitignored throwaway tree and an archive moves with a `data/` copy."""
    if root is not None:
        return Path(root)
    return Path(settings.db_path).parent / "archive"


def blob_path(sha256: str, ext: str, *, root: str | Path | None = None) -> Path:
    """Where a blob lives. Fanned out on the first two hex characters so no directory holds
    more than a few thousand files -- a 30-day round is ~46 GB and ext4 slows badly on a
    single flat directory of that size."""
    return archive_root(root) / "blobs" / sha256[:2] / f"{sha256}.{ext}"


def put(data: bytes, *, root: str | Path | None = None) -> tuple[str, str, int, bool]:
    """Store bytes; return (sha256, mime, n_bytes, is_new).

    `is_new` is False when the blob was already on disk, which is the common case and the
    point of the design -- the caller still writes its own key row, because the same bytes
    reached us under a different (identity, time). NEVER overwrites: identical sha means
    identical content, so a rewrite could only waste I/O.
    """
    sha = hashlib.sha256(data).hexdigest()
    mime, ext = sniff(data)
    path = blob_path(sha, ext, root=root)
    if path.exists():
        return sha, mime, len(data), False
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name in the SAME directory, then rename. A crash mid-write would
    # otherwise leave a short file at a name that claims to be a full sha256, and every
    # later run would trust it -- silent corruption that no checksum ever revisits.
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(path)
    return sha, mime, len(data), True


def get(sha256: str, ext: str, *, root: str | Path | None = None) -> bytes:
    """Read one blob back. Raises FileNotFoundError if the index references bytes the disk
    does not have -- a loud failure, because serving a replay from a half-copied archive is
    exactly the silent-degradation class this whole design exists to prevent."""
    return blob_path(sha256, ext, root=root).read_bytes()


def exists(sha256: str, ext: str, *, root: str | Path | None = None) -> bool:
    return blob_path(sha256, ext, root=root).exists()


def ext_for_mime(mime: str) -> str:
    """The extension a mime maps to, for rebuilding a path from an index row."""
    for _magic, m, ext in _SIGNATURES:
        if m == mime:
            return ext
    return {"video/mp4": "mp4"}.get(mime, "txt")
