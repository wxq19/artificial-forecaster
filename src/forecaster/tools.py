"""Agent-facing tools.

The model can only emit a tool CALL (structured JSON like
{"name": "query_obs", "station": "KORD", ...}); it never sees SQL or a
connection. This module validates the call and runs it against a READ-ONLY
DuckDB connection, so a hallucinated tool call physically cannot write or delete.
Registered here: the read tools plus the output sinks (emit_taf,
submit_taf_worksheet, check_taf) — that's the menu the model is limited to.
The read-only-connection guarantee above covers every DB tool.
Results come back as compact text the VLM can reason over.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from forecaster import (
    awc, charts, imagery, modeldata, neighbors, soundings, store, tafgen,
    tafparse, tafstate, terrain, upper_air_sites, worksheet, wxmaps,
)
from forecaster.config import settings
from forecaster.tafgen import TafProduct
from forecaster.worksheet import TafWorksheet


@dataclass
class ToolResult:
    """What a tool hands back to the loop: a REQUIRED text receipt (a tool reply
    must be text in the OpenAI format) plus any rendered PNGs. Charts reach the
    model as images by the agent loop, which wraps each PNG in a follow-up user
    message. `images` is a list so one call can return several charts (v2)."""

    text: str
    images: list[bytes] = field(default_factory=list)
    videos: list[bytes] = field(default_factory=list)   # mp4 loops (video-capable models only)
    window: tuple | None = None   # (start, end) for time-bounded tools (Fix 3 guard)
    taf: TafProduct | None = None   # emit_taf hands back the captured forecast object
    worksheet: TafWorksheet | None = None   # submit_taf_worksheet hands back the accepted worksheet
    findings: list[str] = field(default_factory=list)   # validate() findings (worksheet/check_taf)

QUERY_OBS = {
    "type": "function",
    "function": {
        "name": "query_obs",
        "description": (
            "Retrieve surface weather observations (METARs) for an airport, oldest "
            "first. Two ways to set the window: for RECENT/trend data give `hours` "
            "(look-back from the most recent observation) — this anchors on the "
            "latest ob SERVER-SIDE, the same anchor get_trend uses, so windows stay "
            "aligned; do NOT compute dates yourself. For a specific historical range "
            "give absolute `start` and `end` (ISO UTC). Each row gives time, wind, "
            "visibility (statute miles), ceiling (ft AGL), present weather, "
            "temperature/dewpoint (C), and altimeter. Do not invent observations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {
                    "type": "string",
                    "description": "4-letter ICAO identifier, e.g. KORD",
                },
                "hours": {
                    "type": "integer",
                    "description": "Relative look-back in hours from the latest ob "
                    "(use for recent/trend questions; aligns with get_trend)",
                },
                "start": {
                    "type": "string",
                    "description": "Absolute UTC start, ISO 8601 (use with end for a "
                    "specific historical range), e.g. 2024-01-12T00:00",
                },
                "end": {
                    "type": "string",
                    "description": "Absolute UTC end, ISO 8601, e.g. 2024-01-13T00:00",
                },
            },
            "required": ["station"],
        },
    },
}

GET_LATEST = {
    "type": "function",
    "function": {
        "name": "get_latest_obs",
        "description": (
            "Most recent observation(s) for an airport, newest first. Use this "
            "when asked about current conditions or 'right now' and NO explicit "
            "time range is given; use query_obs when a date/time range is given."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {
                    "type": "string",
                    "description": "4-letter ICAO identifier, e.g. KORD",
                },
                "n": {
                    "type": "integer",
                    "description": "How many recent obs to return (default 1)",
                },
            },
            "required": ["station"],
        },
    },
}

GET_TREND = {
    "type": "function",
    "function": {
        "name": "get_trend",
        "description": (
            "Render a meteogram (PNG) of how conditions have CHANGED over the last "
            "N hours at an airport, anchored on the most recent observation. The "
            "image stacks temperature/dewpoint, wind, visibility, ceiling, pressure, "
            "and a colored present-weather band over a shared UTC time axis. Use it "
            "to judge whether conditions are improving, deteriorating, or steady "
            "(e.g. for a persistence forecast). Use query_obs for a specific "
            "date/time range; get_latest_obs for a single current ob."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {
                    "type": "string",
                    "description": "4-letter ICAO identifier, e.g. KORD",
                },
                "hours": {
                    "type": "integer",
                    "description": "Look-back window in hours from the latest ob "
                    "(default 24, max 48)",
                },
            },
            "required": ["station"],
        },
    },
}

GET_SOUNDING = {
    "type": "function",
    "function": {
        "name": "get_sounding",
        "description": (
            "Fetch an observed upper-air skew-T sounding (radiosonde) to judge vertical "
            "structure: stability/CAPE, inversions, moisture layers, freezing level, and "
            "wind shear with height. Soundings come from upper-air sites (NOT every "
            "airport), so `site` is an upper-air station id -- a 3-letter id like OUN/MPX "
            "for spc, or a WMO number like 72649/87155 for wyoming and bufr. Most sites "
            "launch at 00Z and/or 12Z, but many also fly OFF-CYCLE special ascents; the "
            "bufr source looks up what was ACTUALLY launched and gives you the most recent "
            "one, which spc/wyoming (fixed 00/12Z images) cannot. Use bufr outside the "
            "United States -- the spc/wyoming image archives have little coverage there."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "site": {
                    "type": "string",
                    "description": ("Upper-air sounding site id (OUN/MPX for spc, a WMO "
                                    "number like 87155 for wyoming/bufr) -- OR, for bufr, "
                                    "just your station's ICAO: it resolves to the nearest "
                                    "radiosonde and tells you which one and how far."),
                },
                "source": {
                    "type": "string",
                    "enum": ["spc", "wyoming", "bufr"],
                    "description": ("Provider: spc (default, richest analysis, US), "
                                    "wyoming (US image archive), or bufr (global, "
                                    "off-cycle-aware, rendered from the raw ascent)"),
                },
            },
            "required": ["site"],
        },
    },
}

# Menu string generated from the catalog so the tool contract can't drift from wxmaps.
_MAP_MENU = "; ".join(f"{n} ({s.label})" for n, s in wxmaps.CATALOG.items())
GET_MAP = {
    "type": "function",
    "function": {
        "name": "get_map",
        "description": (
            "Fetch a surface or upper-air weather chart as an image for synoptic "
            "situational awareness: fronts and pressure systems, jet stream, steering "
            "flow, moisture, and how the pattern is forecast to evolve. Analysis charts "
            "(surface_*, ocean_*, meso_*) show CURRENT conditions; gfs_* are GFS "
            "FORECAST panels -- for those, pass `fhr`, the forecast hour (a multiple of "
            "6, e.g. 0, 6, 12, 24, 36); averaged-field charts (gfs_mslp_precip) start at "
            "f006, not f000. Charts: " + _MAP_MENU + "."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chart": {
                    "type": "string",
                    "enum": list(wxmaps.CATALOG),
                    "description": "Which chart to fetch (see the list in the description)",
                },
                "fhr": {
                    "type": "integer",
                    "description": "GFS forecast hour, multiple of 6 (0-384); only used "
                    "by the gfs_* forecast charts, ignored otherwise",
                },
            },
            "required": ["chart"],
        },
    },
}

GET_FCST_SOUNDING = {
    "type": "function",
    "function": {
        "name": "get_fcst_sounding",
        "description": (
            "MODEL FORECAST sounding for an airport at a chosen forecast hour -- the "
            "PREDICTED vertical structure (stability/CAPE, inversions, moisture, wind shear) "
            "at a future valid time. Unlike get_sounding, which is an OBSERVED sounding at "
            "00/12Z, this projects the atmosphere forward. `station` is a 4-letter ICAO; "
            "`model` is gfs (default, global), hrrr (CONUS only, high resolution) or ifsoper "
            "(ECMWF, global -- good aloft but it carries no 950/900 mb level, so it resolves "
            "the boundary layer poorly, which is where ceilings and inversions live). `fhr` "
            "is the forecast hour (0 = analysis), snapped to the archived 3-hourly grid. "
            "`form` selects how you receive it: 'chart' (default) is a skew-T image, 'table' "
            "is the numbers level by level, 'both' returns each. Ask for whichever you read "
            "more reliably."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KMSP"},
                "model": {"type": "string", "enum": list(modeldata.PROFILE_MODELS),
                          "description": "forecast model (default gfs)"},
                "fhr": {"type": "integer",
                        "description": "forecast hour (0=analysis; e.g. 6, 12, 24, 36)"},
                "form": {"type": "string", "enum": ["chart", "table", "both"],
                         "description": "skew-T image, per-level numbers, or both "
                                        "(default chart)"},
            },
            "required": ["station"],
        },
    },
}

GET_POINT_FORECAST = {
    "type": "function",
    "function": {
        "name": "get_point_forecast",
        "description": (
            "Hourly MODEL point forecast TABLE for an airport, from ONE model: surface "
            "conditions over time -- temperature, dewpoint, wind, GUST, MSL pressure "
            "(hPa and inHg), cloud cover, visibility and ceiling at each forecast hour. Use "
            "it to see how conditions EVOLVE hour by hour at a site (complements "
            "get_fcst_sounding, the vertical profile at one hour, and get_model_state, which "
            "puts several models side by side). Each row is one forecast hour; read a column "
            "downward for a variable's trend. `station` 4-letter ICAO; `model` is gfs "
            "(default, global), hrrr or nbm (CONUS only) or ifsoper (ECMWF, global, no "
            "gust/visibility/ceiling); `hours` limits the horizon (default 48)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KMSP"},
                "model": {"type": "string", "enum": list(modeldata.MODELS),
                          "description": "forecast model (default gfs)"},
                "hours": {"type": "integer",
                          "description": "forecast hours to include from the run (default 48)"},
            },
            "required": ["station"],
        },
    },
}

GET_CLIMO = {
    "type": "function",
    "function": {
        "name": "get_climo",
        "description": (
            "Retrieve the TYPICAL (climatological) weather for an airport in a given "
            "month, built from ~20 years of observations -- NOT current conditions. Use "
            "it to anchor a forecast to what is normal: sanity-check a TX/TN against the "
            "monthly percentile band, time the diurnal wind shift, and judge fog/stratus "
            "and thunderstorm risk by hour. For what is happening NOW or recently, use "
            "get_latest_obs / query_obs / get_trend instead. Returns daily max/min "
            "temperature normals and records, an hourly diurnal table (temp, wind, gust, "
            "prevailing direction), restriction and thunder/fog frequencies, and altimeter "
            "range. `station` is a 4-letter ICAO; `month` (1-12) defaults to the month of "
            "the station's latest stored observation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {
                    "type": "string",
                    "description": "4-letter ICAO identifier, e.g. KLSV",
                },
                "month": {
                    "type": "integer",
                    "description": "Calendar month 1-12 (default: month of the latest ob)",
                },
            },
            "required": ["station"],
        },
    },
}

# Enums generated from the imagery catalogs so the tool contract can't drift (like get_map).
_SAT_REGION_MENU = ", ".join(imagery.SAT_REGIONS)
_RADAR_REGION_MENU = ", ".join(imagery.RADAR_REGIONS)
_IMG_PRODUCTS = list(imagery.SAT_PRODUCTS) + list(imagery.RADAR_PRODUCTS)
_IMG_REGIONS = list(imagery.SAT_REGIONS) + [
    r for r in imagery.RADAR_REGIONS if r not in imagery.SAT_REGIONS
]
GET_IMAGERY = {
    "type": "function",
    "function": {
        "name": "get_imagery",
        "description": (
            "Fetch OBSERVED satellite or radar imagery as an image for spatial awareness "
            "-- cloud extent and erosion, stratus/fog footprint, convective/cloud-top "
            "structure, moisture, and precipitation coverage. Set `kind`: 'satellite' "
            "(geostationary imagery -- GOES over the Americas, Himawari over the W Pacific/"
            "E Asia, Meteosat over Europe/Africa/Middle East; `product` defaults to geocolor, "
            "also infrared, water_vapor. For a specific airport give its `station` "
            "ICAO and the tool picks the sector that covers it -- do NOT guess a `region`; use "
            "`region` only for a broad or named area) or 'radar' (NEXRAD reflectivity, CONUS "
            "only; give a `station` ICAO for "
            "the local view, a `region` for a mosaic, or set product national_mosaic for "
            "broad context). Radar auto-degrades to a regional or national mosaic when "
            "no credible radar is near the station, and says so in the receipt. Imagery is "
            "NOT truth at the field and not a forecast -- pair it with METARs/trend/model. "
            f"Satellite regions: {_SAT_REGION_MENU}. Radar regions: {_RADAR_REGION_MENU}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["satellite", "radar"],
                    "description": "'satellite' or 'radar'",
                },
                "product": {
                    "type": "string",
                    "enum": _IMG_PRODUCTS,
                    "description": "satellite: geocolor (default)/infrared/water_vapor; "
                    "radar: station_reflectivity/regional_mosaic/national_mosaic",
                },
                "region": {
                    "type": "string",
                    "enum": _IMG_REGIONS,
                    "description": "named area; preferred for satellite (default "
                    "conus_east) and for a radar mosaic",
                },
                "station": {
                    "type": "string",
                    "description": "4-letter ICAO, e.g. KLSV; for radar = the station-local "
                    "view, for satellite = auto-pick the covering sector",
                },
            },
            "required": ["kind"],
        },
    },
}

GET_LOOP = {
    "type": "function",
    "function": {
        "name": "get_loop",
        "description": (
            "Fetch a short satellite LOOP (a time sequence of frames) over an airport, to see "
            "MOTION and TREND that a single still cannot show -- which way cloud is moving and "
            "how fast, whether convection is building or collapsing, whether fog/stratus is "
            "spreading or burning off, and where a boundary is going. Returns a labeled "
            "filmstrip image (oldest to newest) and, for video-capable models, a short video. "
            "Give the `station` ICAO; optionally `product`, `frames` (2-10, default 6), and "
            "`step_min` (minutes between frames, default 30) -- frames x step_min sets how far "
            "back the loop reaches, so 6 x 30 covers 2.5h of trend and 10 x 10 shows the last "
            "90 min in detail. Products: geocolor (default; true colour by day, IR-blended at "
            "night), infrared (cloud-top temperature -- use at night and to spot deepening "
            "convection), water_vapor (mid-level moisture and flow, works day or night). "
            "All three work day or night. GOES (Americas) and Himawari "
            "(Japan/W Pacific) loops cover the station's sector; Meteosat (Europe/Africa/"
            "Middle East) loops are station-centered."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KWRI"},
                "product": {"type": "string", "enum": list(imagery.SAT_PRODUCTS),
                            "description": "geocolor (default)/infrared/water_vapor"},
                "frames": {"type": "integer", "description": "number of frames, 2-10 (default 6)"},
                "step_min": {"type": "integer",
                             "description": "minutes between frames (default 30)"},
            },
            "required": ["station"],
        },
    },
}

GET_NEARBY_OBS = {
    "type": "function",
    "function": {
        "name": "get_nearby_obs",
        "description": (
            "Return the latest surface observation (METAR) from neighbor airfields AROUND "
            "this station -- the mesoscale picture for upstream advection, frontal position, "
            "and whether a restriction (fog, low ceiling, gusts) is regional or purely local. "
            "Each neighbor is labeled with its distance, compass bearing FROM your station, and "
            "elevation difference, so you can reason about what is upwind/upslope. Best used "
            "AFTER get_terrain: look at that map, then pass `stations` here to pull obs for the "
            "specific fields you care about (e.g. the ones upwind or toward a coast); omit it "
            "to get the nearest few. Neighbors are read from the same observation store as your "
            "own obs (never live), pre-filtered to before your cutoff."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KWRI"},
                "stations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific neighbor ICAOs to fetch (from the get_terrain map's "
                                   "blue dots / fetchable list). Omit to get the nearest n.",
                },
                "n": {
                    "type": "integer",
                    "description": "How many nearest neighbors to return when `stations` is "
                                   "omitted (default 5, max 5)",
                },
            },
            "required": ["station"],
        },
    },
}

GET_TERRAIN = {
    "type": "function",
    "function": {
        "name": "get_terrain",
        "description": (
            "Return the STATIC terrain and coastline around an airport as a text summary "
            "plus a shaded-relief map image -- station elevation, local relief, the "
            "directions terrain rises (upslope) and falls (downslope), the landform "
            "(valley/basin, ridge/exposed, sloped, flat), and the nearest coast (direction "
            "and distance). The relief map also PLOTS the nearby airfields at their true "
            "positions: blue dots (with labels) are stations you can pull observations for "
            "via get_nearby_obs, violet dots are context for orientation. Use this FIRST to "
            "orient on your surroundings, then decide which neighbor obs to fetch. Anticipate "
            "terrain-driven weather: upslope fog/precipitation, downslope drying/warming, "
            "cold-air pooling in valleys, and sea-breeze or advection fog near a coast. "
            "Geography only -- it never changes and is not a forecast; combine it with the "
            "obs, trend, and model data. NOTE: the coast check sees OCEAN only, so large "
            "inland lakes are not flagged."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KVBG"},
            },
            "required": ["station"],
        },
    },
}

GET_PREVIOUS_TAF = {
    "type": "function",
    "function": {
        "name": "get_previous_taf",
        "description": (
            "Return the PREVIOUS official TAF for this airport -- the human forecast that "
            "was in effect just before your issue time -- for continuity: what the last "
            "forecaster expected and whether conditions have since diverged. Read from the "
            "archive (NOT live) and pre-filtered to before your cutoff, so it is never the "
            "forecast you are being compared against. Returns raw text plus a decoded "
            "per-period summary. Reason independently; do not copy it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KWRI"},
            },
            "required": ["station"],
        },
    },
}

GET_CURRENT_TAF = {
    "type": "function",
    "function": {
        "name": "get_current_taf",
        "description": (
            "Fetch the CURRENT official TAF for an airport (live from aviationweather.gov) "
            "so you can compare the issued forecast to your own reasoning -- continuity, "
            "what the previous forecaster expected, and whether an amendment is warranted. "
            "Returns the raw TAF text and a decoded per-period summary. This is the human "
            "product, not truth; your job is to reason independently, not copy it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KBLV"},
            },
            "required": ["station"],
        },
    },
}

CHECK_TAF = {
    "type": "function",
    "function": {
        "name": "check_taf",
        "description": (
            "Run the AFMAN 15-124 rule checker on a candidate TAF WITHOUT emitting it -- an "
            "explicit dry-run of the same validation emit_taf applies. Fill the same fields "
            "as emit_taf (prevailing period + FM/BECMG/TEMPO groups, TX/TN, QNH). Returns "
            "the rendered TAF text plus any rule findings, so you can iterate on structure "
            "before the final emit. Use emit_taf when you are ready to submit."
        ),
        "parameters": TafProduct.model_json_schema(),
    },
}

GET_MODEL_STATE = {
    "type": "function",
    "function": {
        "name": "get_model_state",
        "description": (
            "Multi-model surface forecast table (GFS + HRRR + NBM side by side) for your "
            "station or a pre-fetched neighbor, from archived model runs pinned to your issue "
            "time. Columns: T/Td (C), wind, gust, MSLP, cloud%, vis, ceiling; rows are valid "
            "times. Use it to see where the models AGREE or DISAGREE on the surface evolution "
            "(e.g. peak gust timing). HRRR is CONUS-only + ~48h; NBM is a govt multi-model "
            "BLEND (a consensus baseline, not an independent ingredient)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KMSP"},
                "location": {"type": "string", "description": (
                    "Optional pre-fetched point to read instead of the station (a neighbor "
                    "ICAO or grid id); defaults to the station.")},
                "model": {"type": "string", "enum": list(modeldata.MODELS),
                          "description": "Optional single model; default shows every archived model."},
                "hours": {"type": "integer", "description": "Optional cap on forecast hours shown (1-48)."},
            },
            "required": ["station"],
        },
    },
}

GET_HAZARD_SCAN = {
    "type": "function",
    "function": {
        "name": "get_hazard_scan",
        "description": (
            "Cross-model ICING + TURBULENCE diagnosis from GFS + HRRR pressure-level fields at "
            "one valid time (no model has a native icing/turbulence field, so conditions are "
            "diagnosed and confirmed across models). Reports per-level T/RH (+ GFS cloud-liquid) "
            "for supercooled-icing potential, plus CAPE/omega/deep-layer shear for convective "
            "and mechanical/CAT turbulence, each with a cross-model agreement note. The flags "
            "are a stated rule over the raw values, not a verdict -- reason over the evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KMSP"},
                "location": {"type": "string", "description": (
                    "Optional pre-fetched point (site or grid id); defaults to the station. "
                    "Hazards are pre-fetched for the site + grid only, not neighbor airfields.")},
                "valid_time": {"type": "string", "description": (
                    "Optional ISO valid time, e.g. 2026-07-17T21:00Z; snaps to the nearest "
                    "stored step. Defaults to the earliest forecast hour with hazard-level "
                    "data.")},
            },
            "required": ["station"],
        },
    },
}

GET_MODEL_VERIFICATION = {
    "type": "function",
    "function": {
        "name": "get_model_verification",
        "description": (
            "How the recent model RUNS scored against OBSERVED METARs in the hours leading up "
            "to your issue time. One block per field (temperature, dewpoint, altimeter, wind "
            "direction, wind speed, gust): rows are valid hours, columns are successive model "
            "runs, so you can read across a row to see whether the fresher run was closer. "
            "Each run gets a mean signed error (a bias to subtract) and a typical error size "
            "(how far off to expect any single hour). Use it to weight raw model output -- "
            "e.g. to back off a gust or temperature the model has been consistently missing. "
            "Reads obs already in the store (leakage-safe), so it only covers hours at or "
            "before your issue time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KMSP"},
                "model": {"type": "string", "enum": list(modeldata.MODELS),
                          "description": "Optional single model; default shows every archived model."},
            },
            "required": ["station"],
        },
    },
}

GET_NEARBY_MODEL_DATA = {
    "type": "function",
    "function": {
        "name": "get_nearby_model_data",
        "description": (
            "One model field's value at ALL pre-fetched points around your station (the site, "
            "neighbor airfields, and a coarse upstream grid) at one valid time -- for gradient "
            "and advection reasoning (e.g. is colder/moister air upstream?). Pick a variable "
            "ALIAS: surface t2m, td2m, gust, mslp, vis, ceil, tcdc (wind is u10/v10 for GFS/HRRR "
            "or wind/wdir for NBM). Values convert to friendly units where known."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KMSP"},
                "variable": {"type": "string", "description": (
                    "Field alias, e.g. t2m (2m temp), mslp, gust, vis.")},
                "model": {"type": "string", "enum": list(modeldata.MODELS),
                          "description": "Model to read (default gfs)."},
                "valid_time": {"type": "string", "description": (
                    "Optional ISO valid time; snaps to the nearest stored step. Defaults to "
                    "the earliest forecast hour.")},
            },
            "required": ["station", "variable"],
        },
    },
}

GET_ENSEMBLE_PROB = {
    "type": "function",
    "function": {
        "name": "get_ensemble_prob",
        "description": (
            "GEFS ENSEMBLE probabilities for your station -- how the 31-member spread turns into "
            "the CHANCE of each condition per hour. Ceiling and visibility are given as the "
            "percent of members in each TAFVER category; wind speed and gust as the percent of "
            "members at or above 15/25/35/45 kt; temperature and dewpoint as the p10/p50/p90 "
            "spread. Use it to judge CONFIDENCE, not a single value -- e.g. whether a restriction "
            "or a gust is likely enough to put in the prevailing group or belongs in a TEMPO/PROB. "
            "Only available where a GEFS ensemble was pre-fetched."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KMSP"},
            },
            "required": ["station"],
        },
    },
}

TOOLS = [QUERY_OBS, GET_LATEST, GET_TREND, GET_SOUNDING, GET_MAP, GET_FCST_SOUNDING,
         GET_POINT_FORECAST, GET_CLIMO, GET_IMAGERY, GET_LOOP, GET_NEARBY_OBS, GET_TERRAIN,
         GET_MODEL_STATE, GET_HAZARD_SCAN, GET_MODEL_VERIFICATION, GET_NEARBY_MODEL_DATA,
         GET_ENSEMBLE_PROB, GET_CURRENT_TAF, CHECK_TAF]

# The OUTPUT tool: the model emits its forecast as the fields of a TafProduct, and
# our code renders + checks it. The parameter schema IS the pydantic model's JSON
# schema, so the one class is both the tool contract and the validator. Unlike the
# read tools, emit_taf is a SINK -- its result is the AFMAN check, not data to
# reason over -- so the loop can feed validate() findings back for a re-emit.
EMIT_TAF = {
    "type": "function",
    "function": {
        "name": "emit_taf",
        "description": (
            "Emit a complete Air Force terminal aerodrome forecast (TAF) as "
            "structured fields. Fill the prevailing period and any FM/BECMG/TEMPO "
            "change groups. Rules: a routine TAF is valid 30 hours; visibility is in "
            "METERS (9999 = unrestricted, >=7SM); wind direction is degrees to the "
            "nearest 10 as an INTEGER (or 'VRB'); QNH is the altimeter in inches of "
            "mercury (e.g. 29.92); include CB cloud type whenever a thunderstorm (TS) "
            "is forecast; do not put QNH in a TEMPO group. Every AF TAF must include a "
            "max (TX) and min (TN) temperature, each as "
            '{"temp_c": <Celsius>, "day": <1-31>, "hour": <0-23 UTC>}. For clear skies '
            "pass an EMPTY clouds list (it renders SKC); SKC/CLR are not valid cloud "
            "cover values. Base the forecast only on tool-provided data."
        ),
        "parameters": TafProduct.model_json_schema(),
    },
}

# The WORKSHEET sink: the model submits its pre-emit reasoning as a single validated
# TafWorksheet (schema = the pydantic model's JSON schema, like emit_taf). A SINK, not
# data -- the receipt is the completeness check, so the loop can feed findings back for
# a re-submit. On success the accepted worksheet rides back on ToolResult.worksheet.
SUBMIT_WORKSHEET = {
    "type": "function",
    "function": {
        "name": "submit_taf_worksheet",
        "description": (
            "Submit your pre-forecast reasoning WORKSHEET before emit_taf: a single "
            "structured object capturing the data you reviewed, the current state, the "
            "drivers, hazards, a forecast timeline, your sanity checks (cross-check each "
            "TX/TN against the observed diurnal temperature range, and take any hPa->inHg "
            "pressure value from the inHg column the model tools already print rather than "
            "converting it by hand), the TAF strategy, uncertainty, and "
            "a final assessment. It returns a completeness check -- correct any findings "
            "and re-submit until clean, THEN emit the TAF from your timeline and strategy. "
            "Fill it ONCE as a single call (reason across your earlier tool calls first)."
        ),
        "parameters": TafWorksheet.model_json_schema(),
    },
}


def _decoded_line(r: dict) -> str:
    """One scannable decoded ob line (no leading indent): time | type | wind | vis |
    ceiling | present-wx | T/Td. Shared by _fmt and the neighbor renderer."""
    wind = "—"
    if r["wind_speed"] is not None:
        d = (
            f"{r['wind_dir_deg']:03d}"
            if r["wind_dir_deg"] is not None
            else (r["wind_dir_card"] or "VRB")
        )
        g = f"G{r['wind_gust']}" if r["wind_gust"] else ""
        wind = f"{d}/{r['wind_speed']}{g}"
    vis = "—" if r["vis_sm"] is None else f"{(r['vis_flag'] or '')}{r['vis_sm']:g}SM"
    ceil = "unlim" if r["ceiling_ft"] is None else f"{r['ceiling_ft']}ft"
    wx = " ".join(r["weather"]) or "-"
    t = "—" if r["temp_c"] is None else f"{r['temp_c']}"
    dp = "—" if r["dewpoint_c"] is None else f"{r['dewpoint_c']}"
    kind = r["report_type"] or "—"
    return (
        f"{r['obs_time']:%Y-%m-%dT%H:%MZ} {kind:<5} {wind:<11} "
        f"{vis:<7} {ceil:<7} {wx:<14} {t}/{dp}"
    )


def _fmt(rows: list[dict], order: str = "oldest first") -> str:
    """Per ob: a decoded summary line (our normalized vis_sm/ceiling_ft) followed
    by the RAW METAR beneath it, so nothing the decoder skips — RMK, RVR, exact
    pressure, peak wind — is lost to the model. The raw line is the ground truth;
    the decoded line is a scannable convenience. `order` only labels the header to
    match how the caller sorted the rows (range reads run oldest-first; a 'latest'
    read stays newest-first — the sort order carries intent, so we don't flatten it)."""
    if not rows:
        return "(no observations in range)"
    out = [
        f"{len(rows)} observations (UTC, {order}). Each ob: decoded summary, "
        "then the raw METAR/SPECI beneath. A SPECI means weather forced an "
        "off-cycle report — treat it as a significance signal.",
        "decoded cols: UTC time (ISO) | type | wind | vis | ceiling | present-wx | T/Td(C)",
    ]
    for r in rows:
        out.append(f"  {_decoded_line(r)}")
        out.append(f"    {r['raw']}")
    return "\n".join(out)


def _window_line(start, end) -> str:
    """Canonical window echo so every time-bounded result states its exact span in
    one comparable line (Fix 2)."""
    return f"window: {start:%Y-%m-%dT%H:%MZ} .. {end:%Y-%m-%dT%H:%MZ}"


def _resolve_window(con, station, args):
    """Resolve a query window. Relative mode (preferred for recent/trend): `hours`
    anchors on the latest ob — IDENTICAL to get_trend, so windows align across
    tools. Absolute mode: explicit ISO start+end. The seam owns the arithmetic;
    the model never computes timestamps. Returns (start, end, None) on success, or
    (None, None, reason) — the reason DISTINGUISHES 'no obs for this station' from
    'no window arguments given' so the model gets accurate feedback (#9)."""
    if args.get("hours") is not None:
        anchor = store.latest(con, station, 1)
        if not anchor:
            return None, None, f"no observations stored for {station} to anchor a relative window"
        end = anchor[0]["obs_time"]
        return end - timedelta(hours=_int_arg(args["hours"], 24, lo=1)), end, None
    if args.get("start") and args.get("end"):
        # Normalize to NAIVE UTC: fromisoformat('...Z') yields a tz-AWARE datetime,
        # which would compare unequal to get_trend's naive obs_time window and
        # false-trip the guard. The seam owns the naive-UTC contract (see store).
        return _naive_utc(args["start"]), _naive_utc(args["end"]), None
    return None, None, ("give either hours (relative to the latest ob) or both start "
                        "and end (ISO UTC)")


def _naive_utc(iso: str) -> datetime:
    """Parse an ISO datetime to naive UTC (drop any 'Z'/offset)."""
    dt = datetime.fromisoformat(iso)
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _int_arg(v, default: int, *, lo: int, hi: int | None = None) -> int:
    """Coerce a model-supplied count/duration to an int and clamp. Models emit numbers
    as strings; a value that will not parse raises ValueError, which run_tool turns into
    tool feedback rather than a dead loop."""
    n = default if v is None else int(v)
    n = max(lo, n)
    return n if hi is None else min(n, hi)


_CLEAR_SKY_COVERS = {"SKC", "CLR", "NSC", "NCD"}


def _has_clear_sky_layer(args: dict) -> bool:
    """True if any authored cloud layer uses a clear-sky token as its cover. A
    CloudLayer has no clear-sky value (clear = an empty clouds list), and its required
    height_ft can trip first and mask the cover mistake, so we detect it from the raw
    args to attach the right hint regardless of which schema error fired."""
    periods = [args.get("prevailing") or {}, *(args.get("groups") or [])]
    for period in periods:
        if not isinstance(period, dict):
            continue
        for layer in period.get("clouds") or []:
            if isinstance(layer, dict) and str(layer.get("cover", "")).upper() in _CLEAR_SKY_COVERS:
                return True
    return False


def _taf_schema_error(verb: str, e: ValidationError, args: dict) -> str:
    """Format a TafProduct ValidationError as correctable feedback (shared by emit_taf
    and check_taf). Names the fix for the two shapes the JSON schema hides, so the model
    does not reverse-engineer it from a terse error (which fed the observed rumination)."""
    errs = "\n".join(f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                     for err in e.errors())
    hints = []
    # (a) TafTemp: a None|TafTemp union failure is a terse two-branch error.
    if any(err["loc"] and str(err["loc"][0]) in ("max_temp", "min_temp") for err in e.errors()):
        hints.append('max_temp/min_temp each need three integers -- '
                     '{"temp_c": <Celsius>, "day": <1-31>, "hour": <0-23>}.')
    # (b) Clear-sky token as a cloud cover: a CloudLayer has no clear-sky value, and
    # its required height_ft can mask the cover error -- so detect it from the args.
    if _has_clear_sky_layer(args):
        hints.append("for clear skies pass an EMPTY clouds list [] (renders SKC); "
                     "SKC/CLR/NSC are not valid cloud covers.")
    hint = "".join(f"\n  note: {h}" for h in hints)
    return (f"{verb} rejected ({e.error_count()} schema error(s)); fix and re-{verb.split('_')[0]}:"
            f"\n{errs}{hint}")


def _emit_taf(args: dict) -> ToolResult:
    """Capture the model's structured forecast: build a TafProduct (guardrails fire
    here), render it, and run the AFMAN rule check + round-trip. The receipt is that
    check, phrased so the model can re-emit a fix; the built product rides back on
    ToolResult.taf. A schema/guardrail failure is reported as text, not raised, so a
    malformed call becomes correctable feedback rather than a crashed loop."""
    try:
        product = TafProduct(**args)
    except ValidationError as e:
        return ToolResult(_taf_schema_error("emit_taf", e, args))
    findings = tafgen.validate(product)
    try:
        text = tafgen.render_taf(product)
    except Exception as e:  # noqa: BLE001 -- a group missing required timing; report, don't crash
        return ToolResult(
            f"emit_taf built but could not render ({type(e).__name__}: {e}); a change group "
            "is likely missing its day/hour fields. Fix and re-emit.", taf=product)
    lines = ["TAF emitted:", "", text, ""]
    if findings:
        lines.append(f"AFMAN check found {len(findings)} issue(s) -- correct them and re-emit:")
        lines += [f"  - {f}" for f in findings]
        return ToolResult("\n".join(lines), taf=product)   # skip round-trip on a known-bad TAF
    try:
        diffs = tafgen.roundtrip(product)
    except Exception as e:  # noqa: BLE001 -- a group that renders but won't re-parse is feedback, not a crash
        return ToolResult(
            f"emit_taf passed the AFMAN check but its render could not be parsed back "
            f"({type(e).__name__}: {e}); a change group is likely missing timing fields. "
            "Fix and re-emit.", taf=product)
    lines.append("AFMAN check: clean.")
    if diffs:
        lines.append("round-trip differences: " + "; ".join(diffs))
    return ToolResult("\n".join(lines), taf=product)


def _check_taf(args: dict) -> ToolResult:
    """Dry-run the AFMAN checker on a candidate TAF WITHOUT emitting it: build + render +
    validate() and hand the findings back, but do NOT set ToolResult.taf (a driver
    captures the final TAF only from emit_taf). Same feedback-not-crash contract as
    _emit_taf; lets the model iterate on structure before the final emit."""
    try:
        product = TafProduct(**args)
    except ValidationError as e:
        return ToolResult(_taf_schema_error("check_taf", e, args))
    findings = tafgen.validate(product)
    try:
        text = tafgen.render_taf(product)
    except Exception as e:  # noqa: BLE001 -- a group missing timing renders visibly, doesn't crash
        return ToolResult(
            f"check_taf built but could not render ({type(e).__name__}: {e}); a change group "
            "is likely missing its day/hour fields.", findings=findings)
    lines = ["check_taf (dry run -- not emitted):", "", text, ""]
    if findings:
        lines.append(f"AFMAN check found {len(findings)} issue(s):")
        lines += [f"  - {f}" for f in findings]
    else:
        lines.append("AFMAN check: clean. Ready to emit_taf.")
    return ToolResult("\n".join(lines), findings=findings)


def _get_current_taf(args: dict) -> ToolResult:
    """Fetch the current official TAF from AWC (network, no DB) and hand back the raw text
    plus a decoded per-period summary. A fetch/parse failure becomes feedback, not a crash."""
    station = args.get("station")
    if not station:
        return ToolResult('error: get_current_taf needs a "station" ICAO, e.g. "station": "KBLV"')
    icao = str(station).upper()
    try:
        tafs = awc.fetch_taf(icao)
    except Exception as e:  # noqa: BLE001 -- a fetch failure becomes feedback, not a dead loop
        return ToolResult(f"error: could not fetch TAF for {icao} ({type(e).__name__}: {e})")
    if not tafs:
        return ToolResult(f"No current TAF is available for {icao} from AWC "
                          "(not all airfields issue TAFs).")
    issue, raw = tafs[0]                          # most recent issuance for this station
    lines = [f"Current official TAF for {icao} (issued {issue:%Y-%m-%dT%H:%MZ}, source "
             "aviationweather.gov). This is the human forecast, not truth -- reason "
             "independently.", "", raw, ""]
    try:
        obs = tafparse.parse(raw)
        lines += ["Decoded per-period summary:", tafparse.render(obs)]
    except Exception as e:  # noqa: BLE001 -- the raw is always shown; a decode miss is non-fatal
        lines += [f"(could not decode the TAF: {type(e).__name__}: {e}; the raw text above stands)"]
    return ToolResult("\n".join(lines))


def _get_previous_taf(con, station: str) -> ToolResult:
    """Return the most recent PRIOR official TAF for continuity -- read from the per-run
    archive (NOT live), which the collector pre-loaded with ONLY the pre-cutoff bulletin,
    so it can never be the forecast being scored. Raw + decoded, mirroring get_current_taf."""
    row = store.previous_human_taf(con, station)
    if not row:
        return ToolResult(f"No previous TAF is on file for {station} before your issue time "
                          "(e.g. this is the first cycle collected here). Reason from the data.")
    issue, raw = row["issue_time_utc"], row["raw_taf"]
    lines = [f"Previous official TAF for {station} (issued {issue:%Y-%m-%dT%H:%MZ}; the forecast "
             "in effect before your issue time). Continuity reference, not truth -- reason "
             "independently.", "", raw, ""]
    try:
        obs = tafparse.parse(raw)
        lines += ["Decoded per-period summary:", tafparse.render(obs)]
    except Exception as e:  # noqa: BLE001 -- the raw is always shown; a decode miss is non-fatal
        lines += [f"(could not decode the TAF: {type(e).__name__}: {e}; the raw text above stands)"]
    return ToolResult("\n".join(lines))


def _submit_worksheet(args: dict, *, evidence_ids: list[str] | None = None) -> ToolResult:
    """The worksheet SINK: build a TafWorksheet (guardrails fire), run the semantic
    completeness check, and return findings as the receipt so the model re-submits a fix.
    The accepted (or best-so-far) worksheet rides back on ToolResult.worksheet; findings
    on ToolResult.findings so the driver can gate emit_taf in `required` mode. Mode +
    evidence_mode come from config; `evidence_ids` (threaded by the loop) enables
    evidence-ref RESOLUTION -- None means presence-only. Never raises: a schema failure
    is correctable feedback, exactly like _emit_taf."""
    try:
        ws = TafWorksheet(**args)
    except ValidationError as e:
        errs = "\n".join(f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                         for err in e.errors())
        return ToolResult(f"submit_taf_worksheet rejected ({e.error_count()} schema error(s)); "
                          f"fix and re-submit:\n{errs}")
    findings = worksheet.validate(
        ws, mode=settings.worksheet_mode, evidence_mode=settings.evidence_mode,
        known_evidence_ids=evidence_ids,
    )
    if findings:
        blocking = worksheet.blocking_findings(findings)
        advisory = len(findings) - len(blocking)
        head = (f"Worksheet received. Completeness check found {len(findings)} issue(s) "
                f"({len(blocking)} blocking"
                + (f", {advisory} advisory" if advisory else "") + ") -- address and re-submit:")
        lines = [head] + [f"  - {f}" for f in findings]
        return ToolResult("\n".join(lines), worksheet=ws, findings=findings)
    return ToolResult(
        "Worksheet received. Completeness check: clean. Proceed to emit_taf, deriving the "
        "TAF from your forecast_timeline and taf_strategy.", worksheet=ws, findings=[])


def _fetch_stamp() -> str:
    """UTC wall-clock stamp for a 'latest'-image receipt. STAR/IEM serve the most recent
    frame with no embedded valid-time, so the fetch time is the only cycle marker the model
    (and later per-run drift analysis) has -- it belongs on the receipt's first line."""
    return f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}"


def _resolve_sounding_site(site: str) -> tuple[str, str]:
    """(wmo, note) for a site id. A 4-letter ICAO is resolved to its NEAREST radiosonde via
    upper_air_sites, so the model can ask for a station it knows instead of having to carry a
    WMO number -- and the note states which site it actually got, and how far away, because
    'the sounding near KWRI' is 174 km away and that materially changes how it should be read."""
    s = str(site).strip().upper()
    if s.isdigit():
        return s, ""
    rows = upper_air_sites.sites_for(s)
    if not rows:
        return s, ""
    wmo, name, dist, brg, _la, _lo = rows[0]
    others = ", ".join(f"{w} ({d:.0f} km {b})" for w, _n, d, b, _a, _o in rows[1:])
    lead = (f"{wmo} {name} is on {s}'s own field ({dist:.0f} km)." if dist <= 25.0
            else (f"{s} has no radiosonde of its own; using the nearest, {wmo} {name}, "
                  f"{dist:.0f} km {brg} of the field."))
    return wmo, lead + (f" Others nearby: {others}." if others else "") + " "


def _sounding_bufr(site: str) -> ToolResult:
    """Observed sounding via the BUFR source: look up what was ACTUALLY launched, fetch the
    raw ascent, render it here (charts.py owns matplotlib; soundings.py returns a profile).

    Deliberately does NOT snap to 00/12Z. It asks the provider's inventory for the newest
    launch, so an off-cycle special ascent -- released precisely because something is
    happening -- is served instead of being silently rounded away to the last synoptic hour."""
    site, note = _resolve_sounding_site(site)
    try:
        # resolve_source, NOT latest_time: the provider carries a site under BUFR (~1 s ascent)
        # or FM35 (the TEMP bulletin) and the sets differ. 47646 TATENO -- the nearest
        # radiosonde to RJTY -- 400s under BUFR and has 462 launches under FM35, so a
        # BUFR-only lookup reported Japan as having no upper-air data at all.
        got = soundings.resolve_source(site)
    except Exception as e:  # noqa: BLE001 -- unknown id / provider hiccup -> feedback
        return ToolResult(f"error: could not read the sounding inventory for {site} "
                          f"({type(e).__name__}: {e}); `site` must be a WMO number or a "
                          "station ICAO on the archived roster, e.g. 87155 or KWRI.")
    src, t = got if got else ("BUFR", None)
    if t is None:
        # An empty inventory means EITHER a quiet site OR an id the provider does not know --
        # the scrape returns [] for both. Ask for the whole record before naming a cause: the
        # SPC source takes 3-letter ids (OUN, MPX) that are a different namespace, so a model
        # carrying one over to `bufr` is likely, and "the site is not reporting" would then be
        # a confidently stated falsehood.
        try:
            ever = soundings.last_known_time(site)
        except Exception:  # noqa: BLE001 -- the diagnosis is a nicety; never fail the reply on it
            ever = None
        if ever is None:
            return ToolResult(f"error: site {site} has no launch record at all, so it is "
                              "probably not a radiosonde site id (or the inventory is "
                              "unreadable). `site` here must be a WMO number or a station "
                              "ICAO on the archived roster, e.g. 87155 or KWRI -- the "
                              "3-letter ids the `spc` source takes are a DIFFERENT namespace.")
        return ToolResult(f"no sounding launched at site {site} in the last 48 h (the site is "
                          f"real and the inventory was read successfully -- its most recent "
                          f"ascent on record is {ever:%Y-%m-%dT%H:%MZ}). Try another site, or "
                          "rely on the model soundings (get_fcst_sounding).")
    try:
        prof = soundings.fetch_profile(site, t, src=src)
        img = charts.skewt(prof)
    except Exception as e:  # noqa: BLE001 -- parse/render failure -> feedback, not a crash
        return ToolResult(f"error: could not build the sounding for {site} at "
                          f"{t:%Y-%m-%dT%H:%MZ} ({type(e).__name__}: {e}).")
    off = "" if t.hour in (0, 12) else "  NOTE: this is an OFF-CYCLE (special) ascent."
    # Name the FEED, not just the provider: FM35 is the TEMP bulletin (mandatory + significant
    # levels, ~100-200) and BUFR is the ~1 s ascent (3,000-4,500), so the level count below
    # means something different in each and the receipt must not blur them.
    return ToolResult(
        f"{note}OBSERVED radiosonde skew-T for site {site}, launched {t:%Y-%m-%dT%H:%MZ}"
        f"{off} Rendered from the raw ascent ({len(prof.pres)} levels thinned from "
        f"{prof.n_raw}); surface {prof.pres[0]:.0f} hPa {prof.tmpc[0]:.1f}/"
        f"{prof.dwpc[0]:.1f} C. (source: Wyoming {src}, {prof.url}); image follows.",
        images=[img])


def _get_sounding(args: dict) -> ToolResult:
    """Fetch an observed skew-T image from a public provider (network, no DB) and
    hand it back for the model to read. Site ids live in the provider's namespace,
    so a bad id/date surfaces as a fetch error the model can correct -- not a crash.
    The receipt cites the exact synoptic time + source URL (provenance)."""
    site = args.get("site")
    if not site:
        return ToolResult('error: get_sounding needs a "site" upper-air id, e.g. "site": "OUN"')
    source = str(args.get("source") or "spc").lower()
    if source not in ("spc", "wyoming", "bufr"):
        return ToolResult(f'error: unknown source {source!r}; use "spc", "wyoming" or "bufr"')
    if source == "bufr":
        return _sounding_bufr(str(site))
    t = soundings.synoptic_time()
    note = ""
    try:
        url = soundings.skewt_url(site, t, source=source)
        img = soundings.fetch_skewt(site, t, source=source)
    except Exception as e:  # noqa: BLE001 -- a fetch failure becomes feedback, not a dead loop
        # Forecasters have a backup provider. SPC and Wyoming serve the SAME observed RAOBs,
        # so on an SPC miss retry Wyoming -- but only when a Wyoming id is derivable. Wyoming
        # takes WMO numbers only, and no site->WMO mapping exists here, so the fallback fires
        # only for a numeric (WMO) id; a 3-letter SPC site gets honest feedback instead.
        if source == "spc" and str(site).isdigit():
            try:
                url = soundings.skewt_url(site, t, source="wyoming")
                img = soundings.fetch_skewt(site, t, source="wyoming")
                source = "wyoming"
                note = (f"note: SPC unavailable ({type(e).__name__}: {e}); serving Wyoming "
                        "sounding for the same station/time.\n")
            except Exception as e2:  # noqa: BLE001 -- both providers failed -> feedback
                return ToolResult(
                    f"error: could not fetch sounding for {str(site).upper()} from SPC "
                    f"({type(e).__name__}: {e}) or Wyoming ({type(e2).__name__}: {e2})")
        else:
            tail = ("" if source == "wyoming" or not str(site).isalpha()
                    else f"; SPC failed and no WMO number is known for {str(site).upper()} "
                         "(pass a WMO number to enable the Wyoming fallback)")
            return ToolResult(
                f"error: could not fetch {source} sounding for {str(site).upper()} "
                f"({type(e).__name__}: {e}); the site may have no launch at this synoptic "
                f"time, or the id may be wrong for this provider (SPC: 3-letter site or WMO; "
                f"Wyoming: WMO){tail}")
    receipt = (
        f"{note}Observed skew-T for {str(site).upper()} at {t:%Y-%m-%dT%H:%MZ} "
        f"(source: {source}, {url}); image follows."
    )
    return ToolResult(receipt, images=[img])


def _get_map(args: dict, station: str | None = None) -> ToolResult:
    """Fetch a catalogued surface/upper-air chart image (network, no DB). A forecast
    chart gets its GFS forecast hour snapped to the 6h grid; an unknown chart name or a
    fetch failure comes back as feedback, not a crash. Receipt cites the source URL.

    `station` is threaded by the harness, NOT taken from the model's arguments, so the
    geographic gate cannot be skipped by omitting it. When present, charts that do not
    depict this station's part of the world are WITHHELD and the TT forecast panels are
    fetched on the station's own domain."""
    name = args.get("chart")
    if not name or name not in wxmaps.CATALOG:
        return ToolResult(
            'error: get_map needs a valid "chart"; choose from: ' + ", ".join(wxmaps.CATALOG)
        )
    domain = "us"
    if station:
        located = True
        try:
            lat, lon = awc.station_latlon(str(station).upper())
            allowed, dom, label = wxmaps.charts_for_latlon(lat, lon)
        except Exception:  # noqa: BLE001 -- a lookup failure must not silently open the gate
            allowed, dom, label, located = (), None, "position unavailable", False
        if name not in allowed:
            if not located:
                # Withholding is still right -- but a failed lookup and a genuinely uncovered
                # region reach this branch identically, and reporting the first as the second
                # asserts a geographic fact about a CONUS station that is simply untrue.
                why = ("this station's position could not be looked up just now, so the "
                       "geographic gate cannot confirm the chart covers its region. This is "
                       "a LOOKUP FAILURE, not a statement that no chart covers the station")
                alt = " Retry the call -- the lookup failure is usually transient."
            elif not allowed:
                why = ("no catalogued chart source covers this station at all -- the A/B "
                       "charts are US products and there is no forecast-panel domain for "
                       "this region")
                alt = (" Use get_imagery (satellite) and the model-data tools "
                       "(get_model_state, get_hazard_scan) for synoptic context instead.")
            else:
                why = ("it depicts a different part of the world, so it is withheld rather "
                       "than served as a confidently-labelled picture of the wrong region")
                alt = f" Available here: {', '.join(allowed)}."
            return ToolResult(f"chart {name!r} is not available for {station} "
                              f"({label}): {why}.{alt}")
        domain = dom or "us"
    spec = wxmaps.CATALOG[name]
    fhr = 0
    run = None
    if spec.source == "tt":
        # An averaged-field chart's first frame is f0 (e.g. gfs_mslp_precip starts at
        # f006), so default AND floor fhr at f0 -- a model that omits fhr or passes 0
        # gets a valid first frame, not a "must be a multiple of 6" rejection.
        f0 = spec.params.get("f0", 0)
        fhr = _int_arg(args.get("fhr"), f0, lo=f0, hi=wxmaps.GFS_MAX_FHR)
        fhr -= fhr % wxmaps.GFS_STEP_H          # snap down to the 6h GFS grid
        run = wxmaps.latest_gfs_run()           # resolve once so the receipt and image agree
    try:
        url = wxmaps.map_url(name, fhr=fhr, run=run, domain=domain)
        img = wxmaps.fetch_map(name, fhr=fhr, run=run, domain=domain)
    except Exception as e:  # noqa: BLE001 -- a fetch failure becomes feedback, not a dead loop
        # TT is third-party + URL-fragile; on failure fall back to the closest SPC
        # mesoanalysis ANALYSIS chart so the model keeps upper-air context. The receipt
        # states the degradation loudly -- it is NOW, not the forecast hour requested.
        # GATED ON THE US DOMAIN: SPC mesoanalysis is CONUS-only, so outside it this
        # fallback would answer a failed South American panel with a picture of Kansas.
        fb = (wxmaps.TT_TO_SPC_MESO.get(name)
              if spec.source == "tt" and domain == "us" else None)
        if fb:
            fb_spec = wxmaps.CATALOG[fb]
            try:
                fb_url = wxmaps.map_url(fb)
                fb_img = wxmaps.fetch_map(fb)
            except Exception as e2:  # noqa: BLE001 -- both failed -> feedback
                return ToolResult(f"error: could not fetch chart {name} ({type(e).__name__}: {e}); "
                                  f"SPC fallback {fb} also failed ({type(e2).__name__}: {e2})")
            note = (f"note: forecast panel unavailable (TropicalTidbits {type(e).__name__}: {e}); "
                    f"serving the CURRENT ANALYSIS (SPC mesoanalysis {fb_spec.label}) instead "
                    f"-- this is now, not the f{fhr:03d} forecast you asked for.")
            return ToolResult(
                f"{note}\n{fb_spec.label} [{fb}] (source: {fb_spec.source}, {fb_url}); "
                "image follows.", images=[fb_img])
        return ToolResult(f"error: could not fetch chart {name} ({type(e).__name__}: {e})")
    lead = f", GFS f{fhr:03d} run {run:%Y-%m-%dT%H:%MZ}" if spec.source == "tt" else ""
    dom = f", domain {domain}" if spec.source == "tt" else ""
    # A substituted panel is announced under what it ACTUALLY shows, not the catalogue
    # label, so the model never reasons about a field it was not given.
    label = spec.label
    if spec.source == "tt":
        var = wxmaps.tt_variant(name, domain)
        if var.field != spec.params["field"]:
            label = f"{var.label}  [substituted for {spec.label} -- unavailable on this domain]"
    return ToolResult(
        f"{label} [{name}]{lead}{dom} (source: {spec.source}, {url}); image follows.",
        images=[img],
    )


def _imagery_satellite(region: str | None, product: str | None,
                       station: str | None) -> ToolResult:
    """Fetch a GOES still from the STAR CDN. An explicit region wins; otherwise a
    `station` ICAO is routed to its covering sector (like radar), so the model need not
    guess which sector sees the field; else default conus_east. Product defaults to
    geocolor (day/night blended -- no night-visible failure)."""
    picked_for = ""
    center: tuple[float, float] | None = None
    icao = ""
    if not region and station:
        icao = str(station).upper()
        try:
            lat, lon = awc.station_latlon(icao)       # live AWC lookup (network, no DB)
        except Exception as e:  # noqa: BLE001 -- unknown id becomes feedback, not a crash
            return ToolResult(f"error: could not resolve a location for {icao} "
                              f"({type(e).__name__}: {e}); give a satellite `region`: "
                              + ", ".join(imagery.SAT_REGIONS))
        region = imagery.satellite_region_for_latlon(lat, lon)
        if region is None:
            return ToolResult(
                f"no geostationary satellite coverage for {icao} (outside the GOES/Himawari/"
                "Meteosat footprints -- e.g. mid-ocean or polar). use radar or another data "
                "tool for this location.")
        center = (lat, lon)
        picked_for = f" (covering sector for {icao})"
    region = region or "conus_east"
    if region not in imagery.SAT_REGIONS:
        return ToolResult(f"error: {region!r} is not a satellite region; choose from: "
                          + ", ".join(imagery.SAT_REGIONS))
    product = product if product in imagery.SAT_PRODUCTS else "geocolor"
    # OSPO Japan has no geocolor; its day/night default is enhanced IR, so relabel honestly.
    if imagery.SAT_REGIONS[region].provider == "himawari_ospo" and product == "geocolor":
        product = "infrared"
    if (imagery.SAT_REGIONS[region].provider == "meteosat_eumetsat_wms"
            and not imagery.meteosat_has_product(product)):
        return _meteosat_no_product(product, imagery.SAT_REGIONS[region].label)
    # Meteosat takes an arbitrary bbox, so for a specific station we center a TIGHT local view
    # on it instead of the wide fixed region (the station-crop upgrade). Others use the sector.
    meteosat_point = center is not None and \
        imagery.SAT_REGIONS[region].provider == "meteosat_eumetsat_wms"
    # Resolve the region the SAME way fetch_satellite will, BEFORE naming it. water_vapor is
    # widened to a synoptic scope inside the fetch, so a receipt built from the tight sector
    # names a sector -- and sometimes a satellite -- the delivered image is not.
    region = imagery.synoptic_region(region, product)
    try:
        if meteosat_point:
            img, url = imagery.fetch_meteosat_point(center[0], center[1], product)
            label = f"Meteosat -- centered on {icao}"
        else:
            img, url = imagery.fetch_satellite(region, product)
            label = f"{imagery.SAT_REGIONS[region].label}{picked_for}"
    except Exception as e:  # noqa: BLE001 -- a fetch failure becomes feedback, not a dead loop
        return ToolResult(f"error: could not fetch {product} satellite "
                          f"({type(e).__name__}: {e})")
    receipt = (f"{product} satellite -- {label}, fetched {_fetch_stamp()} "
               f"(source: {imagery.satellite_source(region)}, {url}); research/informational "
               "imagery, not an operational source. image follows.")
    return ToolResult(receipt, images=[img])


def _radar_national(note: str) -> ToolResult:
    try:
        img = imagery.fetch_radar("national")
    except Exception as e:  # noqa: BLE001 -- feedback, not a dead loop
        return ToolResult(f"error: could not fetch national radar ({type(e).__name__}: {e})")
    # fetch_radar silently degrades national from IEM (PNG) to the NWS RIDGE GIF; cite the
    # source that actually produced this image, not always IEM.
    if _image_mime(img) == "image/gif":
        source, url = "NWS RIDGE", imagery.NWS_RIDGE_GIF_URL
    else:
        source, url = "IEM NEXRAD composite", imagery.radar_url("national")
    return ToolResult(f"{note}, fetched {_fetch_stamp()} (source: {source}, {url}); "
                      "image follows.", images=[img])


def _radar_regional(region: str) -> ToolResult:
    label = imagery.RADAR_REGIONS[region][1]
    try:
        url = imagery.radar_url("regional", region=region)
        img = imagery.fetch_radar("regional", region=region)
    except Exception as e:  # noqa: BLE001 -- feedback, not a dead loop
        return ToolResult(f"error: could not fetch {region} radar ({type(e).__name__}: {e})")
    return ToolResult(f"{label} regional radar mosaic, fetched {_fetch_stamp()} "
                      f"(source: IEM NEXRAD composite, {url}); image follows.", images=[img])


def _meteosat_no_product(product: str, label: str) -> ToolResult:
    """Meteosat cannot serve this product -- say so and return NOTHING.

    Same rule as `_radar_no_coverage`: a text-only refusal beats a confidently labelled
    picture of something else. Falling back to geocolour under the water_vapor name is what
    round 1 actually did over Europe, and SHA-256 showed the three products were one image."""
    return ToolResult(
        f"{product} is not available from Meteosat ({label}). EUMETSAT publishes no "
        f"comparable {product} product for this region, and serving a different channel "
        f"under that name would misinform the analysis. No image is returned. Use "
        f"get_imagery product=geocolor or product=infrared here, and the upper-level flow "
        f"from get_map (500 mb / 300 mb panels) for the moisture and jet picture.")


def _radar_no_coverage(icao: str, near: tuple | None, reason: str) -> ToolResult:
    """No usable radar for this station -- say so and return NOTHING.

    This deliberately does NOT fall back to the US national mosaic. Handing a Japanese or
    Patagonian station a labelled picture of the United States is worse than handing it
    nothing: the model cannot tell that the image is irrelevant, and round 1 shipped exactly
    that (RJTY's 'radar' was a CONUS composite, 1,090 km from the nearest WSR-88D). A
    text-only refusal is honest and costs the model one turn."""
    where = (f" The nearest WSR-88D is {near[0]['id']} {near[0]['name']}, {near[1]:,.0f} km away."
             if near else "")
    return ToolResult(
        f"no radar coverage for {icao}: {reason}.{where} The WSR-88D network is US-only, so "
        f"there is no radar product for this site -- use satellite imagery instead "
        f"(get_imagery kind=satellite, or get_loop for motion). No image is returned.")


def _radar_degrade(icao: str, lat: float, lon: float, reason: str,
                   near: tuple | None = None) -> ToolResult:
    """Fall back from a station-local view to the containing REGIONAL mosaic only.
    `reason` (guard miss or a station-fetch failure) is prepended so the receipt is honest.
    Outside the curated regions there is no fallback -- see `_radar_no_coverage`."""
    reg = imagery.radar_region_for_latlon(lat, lon)
    if reg:
        r = _radar_regional(reg)
        if r.images:                                  # regional succeeded
            r.text = (f"{reason}; showing the {imagery.RADAR_REGIONS[reg][1]} regional "
                      f"mosaic instead. {r.text}")
            return r
        return _radar_no_coverage(icao, near, f"{reason}, and the "
                                  f"{imagery.RADAR_REGIONS[reg][1]} regional mosaic "
                                  f"could not be fetched either")
    return _radar_no_coverage(icao, near, reason)


def _radar_for_station(icao: str, product: str | None) -> ToolResult:
    """Radar for a station. An explicit mosaic product is honored directly; otherwise the
    default/station_reflectivity path tries a station-centered composite when a credible
    WSR-88D is within the 150 km guard, and degrades (regional -> national) with an
    ACCURATE reason on either a guard miss or a station-fetch failure."""
    try:
        lat, lon = awc.station_latlon(icao)           # live AWC lookup (network, no DB)
    except Exception as e:  # noqa: BLE001 -- an unknown id becomes feedback, not a crash
        return ToolResult(
            f"error: could not resolve a location for {icao} ({type(e).__name__}: {e}); "
            "give a radar `region` instead: " + ", ".join(imagery.RADAR_REGIONS))

    near = imagery.nearest_radar(lat, lon)
    guard = imagery.RADAR_STATION_GUARD_KM
    reg = imagery.radar_region_for_latlon(lat, lon)

    # OUT OF NETWORK: no credible local radar AND outside every curated region. Refuse for
    # EVERY product, including an explicit national_mosaic -- the station is not in the US
    # radar network at all, so no US product describes it. (An explicit `region` request,
    # which arrives via _get_imagery rather than here, is still honored: that is the model
    # deliberately asking to look at a named US area.)
    if not reg and not (near and near[1] <= guard):
        return _radar_no_coverage(
            icao, near,
            f"{icao} is outside the curated radar regions and has no WSR-88D within the "
            f"{guard:.0f} km local-radar guard")

    # Honor an explicit mosaic choice directly -- do NOT route it through the guard (which
    # would fabricate a distance reason and hand back the wrong product).
    if product == "national_mosaic":
        return _radar_national("national radar mosaic (broad context only)")
    if product == "regional_mosaic":
        if reg:
            return _radar_regional(reg)
        return _radar_no_coverage(icao, near, f"{icao} is outside the curated radar regions")

    # Default / station_reflectivity: a station-centered local view when a radar is credible.
    if near and near[1] <= guard:
        site, dist = near
        try:
            url = imagery.radar_url("station", center=(lat, lon))
            img = imagery.fetch_radar("station", center=(lat, lon))
        except Exception as e:  # noqa: BLE001 -- degrade, don't dead-end (provider hiccup/outage)
            return _radar_degrade(icao, lat, lon,
                                  f"station radar fetch for {icao} failed ({type(e).__name__}: {e})",
                                  near=near)
        receipt = (f"Station-scale radar around {icao}, fetched {_fetch_stamp()} "
                   f"(nearest WSR-88D: {site['id']} {site['name']}, {dist:.0f} km; "
                   f"source: IEM NEXRAD composite, {url}); image follows.")
        return ToolResult(receipt, images=[img])
    reason = (f"nearest WSR-88D to {icao} is {near[1]:.0f} km away (beyond the {guard:.0f} km "
              f"local-radar guard)" if near else f"no radar site found near {icao}")
    return _radar_degrade(icao, lat, lon, reason, near=near)


def _get_loop(args: dict) -> ToolResult:
    """Fetch a short satellite loop for a station and compose it into a filmstrip (image,
    universal) + a short mp4 (video-capable models). Network, no DB. A missing station,
    no-coverage point, or too-few frames comes back as feedback, not a crash."""
    station = args.get("station")
    if not station:
        return ToolResult('error: get_loop needs a "station" ICAO, e.g. "station": "KWRI".')
    icao = str(station).upper()
    product = str(args["product"]).lower() if args.get("product") else "geocolor"
    if product not in imagery.SAT_PRODUCTS:
        product = "geocolor"
    frames = _int_arg(args.get("frames"), imagery.LOOP_DEFAULT_FRAMES,
                      lo=2, hi=imagery.LOOP_MAX_FRAMES)
    step = _int_arg(args.get("step_min"), imagery.LOOP_DEFAULT_STEP_MIN, lo=10, hi=120)
    try:
        lat, lon = awc.station_latlon(icao)       # live AWC lookup (network, no DB)
    except Exception as e:  # noqa: BLE001 -- unknown id becomes feedback, not a crash
        return ToolResult(f"error: could not resolve a location for {icao} "
                          f"({type(e).__name__}: {e}).")
    # Same gate as the still: refuse rather than loop ten frames of the wrong channel.
    reg = imagery.satellite_region_for_latlon(lat, lon)
    if (reg and imagery.SAT_REGIONS[reg].provider == "meteosat_eumetsat_wms"
            and not imagery.meteosat_has_product(product)):
        return _meteosat_no_product(product, imagery.SAT_REGIONS[reg].label)
    try:
        fr, source, coverage = imagery.satellite_loop(lat, lon, product,
                                                      frames=frames, step_min=step)
    except Exception as e:  # noqa: BLE001 -- no coverage / fetch failure -> feedback
        return ToolResult(f"error: could not build a satellite loop for {icao} "
                          f"({type(e).__name__}: {e}).")
    if len(fr) < 2:
        return ToolResult(f"error: only {len(fr)} loop frame(s) available for {icao}; "
                          "cannot show motion.")
    span = f"{fr[0].label} -> {fr[-1].label}"
    # satellite_loop returns FRAMES; the filmstrip/mp4 are composed here, so an archiver can
    # store the frames and let replay re-compose whatever the model asks for.
    tiles = [(f.label, f.data) for f in fr]
    strip = charts.filmstrip(tiles, title=f"{icao} {coverage} loop  {span}")
    mp4 = charts.loop_mp4(tiles)
    receipt = (f"satellite LOOP -- {coverage} for {icao}: {len(fr)} frames, {span} "
               f"(source: {source}); labeled filmstrip image (oldest->newest) and a short "
               "video follow. research/informational imagery, not an operational source.")
    return ToolResult(receipt, images=[strip], videos=[mp4])


def _get_imagery(args: dict) -> ToolResult:
    """Fetch observed satellite or radar imagery (network, no DB). Dispatches on `kind`;
    infers it from the other args if omitted. Radar runs the station-aware degrade
    cascade. A bad region/fetch comes back as feedback, not a crash."""
    kind = str(args.get("kind") or "").lower()
    product = str(args["product"]).lower() if args.get("product") else None
    region = str(args["region"]).lower() if args.get("region") else None
    station = args.get("station")
    if kind not in ("satellite", "radar"):
        # Infer a missing kind so the call isn't a dead end.
        if product in imagery.SAT_PRODUCTS or region in imagery.SAT_REGIONS:
            kind = "satellite"
        elif station or product in imagery.RADAR_PRODUCTS or region in imagery.RADAR_REGIONS:
            kind = "radar"
        else:
            return ToolResult('error: get_imagery needs "kind": "satellite" or "radar".')
    if kind == "satellite":
        return _imagery_satellite(region, product, station)
    if station:
        return _radar_for_station(str(station).upper(), product)
    if region:
        if region not in imagery.RADAR_REGIONS:
            return ToolResult(f"error: {region!r} is not a radar region; choose from: "
                              + ", ".join(imagery.RADAR_REGIONS))
        if region == "national":
            return _radar_national("national radar mosaic (broad context only)")
        return _radar_regional(region)
    if product == "national_mosaic":
        return _radar_national("national radar mosaic (broad context only)")
    return ToolResult('error: radar needs a "station" (ICAO) for the local view or a '
                      '"region" for a mosaic; for broad context set "product": '
                      '"national_mosaic". Radar regions: ' + ", ".join(imagery.RADAR_REGIONS))


def _fmt_profile_table(prof) -> str:
    """A forecast sounding as NUMBERS, level by level -- the `form='table'` rendering. Same
    archived rows the skew-T is drawn from, so the two forms can never disagree."""
    out = [(f"{'Pres mb':>8}{'Hgt m':>8}{'T C':>7}{'Td C':>7}{'RH %':>6}"
            f"{'Wind':>10}")]
    for p, z, t, d, dr, sp in zip(prof.pres, prof.hght, prof.tmpc, prof.dwpc,
                                  prof.drct, prof.sknt):
        # RH back out of T/Td for display only; the archive stores RH and we derive Td.
        rh = 100.0 * (math.exp((17.625 * d) / (243.04 + d))
                      / math.exp((17.625 * t) / (243.04 + t)))
        out.append(f"{p:>8.0f}{z:>8.0f}{t:>7.1f}{d:>7.1f}{rh:>6.0f}"
                   f"{f'{dr:03.0f}/{sp:.0f}':>10}")
    return "\n".join(out)


def _get_fcst_sounding(con, station: str, args: dict) -> ToolResult:
    """Render an ARCHIVED model forecast sounding (DB read, no network). Pressure-level rows
    come from modeldata.prefetch, so this is pinned to the issue time like every other
    model-data tool. A model without a profile, or a forecast hour the archive does not
    cover, comes back as feedback naming what IS available."""
    model = str(args.get("model") or "gfs").lower()
    if model not in modeldata.PROFILE_MODELS:
        return ToolResult(f"error: {model!r} has no vertical profile; choose from "
                          f"{', '.join(modeldata.PROFILE_MODELS)} (NBM is surface-only)")
    form = str(args.get("form") or "chart").lower()
    if form not in ("chart", "table", "both"):
        form = "chart"
    loc = _resolve_md_location(con, station, None)
    if loc is None:
        return ToolResult(f"error: {station} is not a pre-fetched model-data location. "
                          f"{_md_locations_hint(con)}")
    lat, lon, _name = loc
    times = modeldata.profile_valid_times(con, station, model, lat=lat, lon=lon)
    if not times:
        return ToolResult(f"error: no archived {model.upper()} profile for {station}; the "
                          f"pressure-level bundle may not have been pulled for this cycle.")
    fhr = _int_arg(args.get("fhr"), 12, lo=0, hi=384)
    # fhr counts from the ISSUE time, not from the first archived hour: the level grid snaps
    # to a 00Z-anchored 3-hourly grid, so those differ by up to one step.
    issue = store.model_data_as_of(con, model, lat, lon) or times[0]
    want = issue.replace(minute=0, second=0, microsecond=0) + timedelta(hours=fhr)
    valid = min(times, key=lambda t: abs((t - want).total_seconds()))   # snap to the grid
    try:
        prof = modeldata.build_profile(con, station, model, valid, lat=lat, lon=lon)
    except ValueError as e:
        return ToolResult(f"error: {e}")
    off_h = round((valid - want).total_seconds() / 3600)
    snapped = ("" if not off_h else
               f" (you asked for f{fhr:03d}; the archive is 3-hourly, so this is the nearest "
               f"stored hour, {abs(off_h)}h {'later' if off_h > 0 else 'earlier'})")
    ifs_note = ("\nNOTE: IFS carries no 950/900 mb level, so the boundary layer -- where "
                "ceilings and inversions sit -- is resolved coarsely here. Cross-check low "
                "cloud against GFS or HRRR." if model == "ifsoper" else "")
    receipt = (f"{model.upper()} forecast sounding for {prof.station}, valid "
               f"{valid:%Y-%m-%dT%H}Z (f{prof.fhr:03d} of the {prof.run:%Y-%m-%dT%H}Z run), "
               f"{len(prof.pres)} levels{snapped}.{ifs_note}")
    images = [charts.skewt(prof)] if form in ("chart", "both") else []
    if form in ("table", "both"):
        receipt += "\n\n" + _fmt_profile_table(prof)
    if images:
        receipt += "\nSkew-T image follows."
    return ToolResult(receipt, images=images)


def _uv_to_dirspd(u: float, v: float) -> tuple[int, int]:
    """Wind (u, v in m/s) -> (direction deg to nearest 10, speed kt). A presentation of the
    raw vector; the stored point-forecast data keeps the u/v components."""
    spd = round(math.hypot(u, v) * 1.94384)
    d = int(round((270.0 - math.degrees(math.atan2(v, u))) % 360.0 / 10.0) * 10) % 360
    return d, spd


def _get_point_forecast(con, station: str, args: dict) -> ToolResult:
    """ONE model's hourly surface table, read from the archive (DB read, no network).

    Kept as its own tool rather than folded into get_model_state: it is the single most-used
    tool in the suite, the model already knows its shape, and a one-model table is easier to
    read down a column than the multi-model one. Since 2026-07-28 it reads the GRIBStream
    archive instead of BUFKIT, so it gains the gust, visibility and ceiling columns BUFKIT
    never carried -- gust being the model's worst-scoring TAF element."""
    model = str(args.get("model") or "gfs").lower()
    if model not in modeldata.MODELS:
        return ToolResult(f"error: unknown model {model!r}; choose from "
                          f"{', '.join(modeldata.MODELS)}")
    loc = _resolve_md_location(con, station, None)
    if loc is None:
        return ToolResult(f"error: {station} is not a pre-fetched model-data location. "
                          f"{_md_locations_hint(con)}")
    hours = _int_arg(args.get("hours"), 48, lo=1, hi=384)
    return ToolResult(_fmt_model_state(con, station, loc, [model], hours))


_MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"]


def _cd(v, fmt: str = "{:.0f}") -> str:
    """Climo cell: '--' for a NULL (e.g. an all-NULL quantile), else formatted."""
    return "--" if v is None else fmt.format(v)


def _fmt_climo(meta: dict, monthly: dict, hourly: list[dict]) -> str:
    """Render the climatology product as compact text: header (POR + denominator note),
    temperature normals + records, a 3-hourly diurnal table (UTC key, LST label),
    restriction frequencies by 3h block (n_obs-weighted), a phenomena line + TS/fog peak
    hours, and the altimeter range. Climatology is not a time window -- no window line."""
    st = monthly["station"]
    mon = monthly["month"]
    off = meta.get("utc_offset_hours_std")
    lst_note = f"LST = UTC{off:+.0f}" if off is not None else "LST offset unknown"

    def lst(h: int) -> int:
        return int((h + (off or 0)) % 24)

    out = [
        f"Climatology for {st} -- {_MONTH_NAMES[mon]} (typical conditions, NOT current). "
        f"POR {monthly['por_start_year']}-{monthly['por_end_year']} "
        f"({monthly['n_years_used']} yr, {monthly['n_days']} days, "
        f"{monthly['n_obs_routine']} routine obs). {lst_note}. "
        "Frequencies use routine METARs only; temperatures use all obs.",
        "",
        "TEMPERATURE (daily, C):",
        f"  max (TX): mean {_cd(monthly['tx_mean'], '{:.1f}')}  "
        f"p10/p50/p90 {_cd(monthly['tx_p10'])}/{_cd(monthly['tx_p50'])}/{_cd(monthly['tx_p90'])}  "
        f"record {_cd(monthly['tx_record'])} ({monthly['tx_record_date']})",
        f"  min (TN): mean {_cd(monthly['tn_mean'], '{:.1f}')}  "
        f"p10/p50/p90 {_cd(monthly['tn_p10'])}/{_cd(monthly['tn_p50'])}/{_cd(monthly['tn_p90'])}  "
        f"record {_cd(monthly['tn_record'])} ({monthly['tn_record_date']})",
        "",
        "DIURNAL (every 3h; temp C, wind kt):",
        f"  {'UTC':>3} {'LST':>3} {'temp':>5} {'wind':>5} {'p90':>4} {'gust%':>6} "
        f"{'prevail':>8}",
    ]
    by_hour = {h["hour_utc"]: h for h in hourly}
    for h in range(0, 24, 3):
        r = by_hour.get(h)
        if not r:
            continue
        prevail = "--" if r["dir_mode_sector"] is None else \
            f"{r['dir_mode_sector']} {_cd(r['dir_mode_pct'])}%"
        out.append(
            f"  {h:>3} {lst(h):>3} {_cd(r['temp_mean_c'], '{:.0f}'):>5} "
            f"{_cd(r['wind_mean_kt'], '{:.0f}'):>5} {_cd(r['wind_p90_kt'], '{:.0f}'):>4} "
            f"{_cd(r['gust_pct'], '{:.0f}'):>6} {prevail:>8}"
        )

    # Restriction frequencies collapsed to 3h blocks, n_obs-weighted.
    out += ["", "RESTRICTION FREQUENCY (% of routine obs, by 3h UTC block):",
            f"  {'block':>7} {'cig<3k':>7} {'<1k':>5} {'<500':>5} {'vis<3':>6} {'<1':>5}"]
    for h0 in range(0, 24, 3):
        block = [by_hour[h] for h in range(h0, h0 + 3) if h in by_hour]
        if not block:
            continue
        n = sum(b["n_obs"] or 0 for b in block) or 1

        def wavg(key, block=block, n=n):
            return sum((b[key] or 0) * (b["n_obs"] or 0) for b in block) / n

        out.append(
            f"  {h0:02d}-{h0 + 2:02d}Z  {wavg('pct_cig_lt_3000'):>6.1f} "
            f"{wavg('pct_cig_lt_1000'):>5.1f} {wavg('pct_cig_lt_500'):>5.1f} "
            f"{wavg('pct_vis_lt_3'):>6.1f} {wavg('pct_vis_lt_1'):>5.1f}"
        )

    # Phenomena (monthly) + peak hours (from the hourly rows).
    out += ["", "PHENOMENA (% of routine obs, monthly): "
            f"TS {_cd(monthly['pct_ts'], '{:.1f}')}  fog/mist {_cd(monthly['pct_fog'], '{:.1f}')}  "
            f"rain {_cd(monthly['pct_ra'], '{:.1f}')}  snow {_cd(monthly['pct_sn'], '{:.1f}')}  "
            f"fzra/fzdz {_cd(monthly['pct_fzprecip'], '{:.1f}')}"]

    def peak(key):
        cand = [(r[key], r["hour_utc"]) for r in hourly if r[key]]
        return max(cand) if cand else None
    ts_pk, fog_pk = peak("pct_ts"), peak("pct_fog")
    peaks = []
    if ts_pk:
        peaks.append(f"TS peak ~{ts_pk[1]:02d}Z ({ts_pk[0]:.1f}%)")
    if fog_pk:
        peaks.append(f"fog peak ~{fog_pk[1]:02d}Z ({fog_pk[0]:.1f}%)")
    if peaks:
        out.append("  peak hours: " + "; ".join(peaks))

    out.append(
        f"\nALTIMETER (inHg): mean {_cd(monthly['alt_mean'], '{:.2f}')} "
        f"(range {_cd(monthly['alt_min'], '{:.2f}')}-{_cd(monthly['alt_max'], '{:.2f}')})"
    )
    return "\n".join(out)


def _get_climo(con, args: dict) -> ToolResult:
    """Read the climo_* product for a station-month and render it. Reads only the
    climo tables on the read-only conn -- no ingest, no build. A missing/empty table
    (pre-climo DB) or an unbuilt month returns feedback naming the build script, not a
    crash. No ToolResult.window: climatology is not a time window."""
    station = str(args["station"]).upper()
    try:
        meta = store.climo_meta(con, station)
    except Exception:  # noqa: BLE001 -- climo tables don't exist yet on this DB
        return ToolResult(
            "error: no climatology has been built for this database. Build it with "
            "`uv run python scripts/build_climo.py --station <ICAO> --months <M>`."
        )
    month = args.get("month")
    if month is None:
        anchor = store.latest(con, station, 1)
        if not anchor:
            return ToolResult(
                f"error: no observations stored for {station} to pick a default month; "
                "pass an explicit `month` (1-12)."
            )
        month = anchor[0]["obs_time"].month
    else:
        month = _int_arg(month, month, lo=1, hi=12)
    monthly = store.climo_month(con, station, month)
    if meta is None or monthly is None:
        return ToolResult(
            f"error: climatology for {station} month {month} is not built. Build it with "
            f"`uv run python scripts/build_climo.py --station {station} --months {month}`."
        )
    hourly = store.climo_hours(con, station, month)
    return ToolResult(_fmt_climo(meta, monthly, hourly))


def _get_nearby_obs(con, station: str, args: dict) -> ToolResult:
    """Latest ob from neighbor airfields (DB read, leakage-safe -- the per-run DB is already
    cut off at the issue time). Each row is annotated with distance, bearing FROM the home
    station, and elevation delta so the model can reason spatially. By default returns the n
    nearest; pass `stations` (after reading the get_terrain map) to fetch a chosen subset."""
    roster = neighbors.neighbors_of(station)
    if not roster:
        # The roster is built for EVERY archived station, so an empty list no longer means
        # "not built" -- it means the search found no other METAR airfield within 150 km.
        # Say that plainly: it is a real property of the site (SAWG and SAZN in Patagonia
        # have literally none), and a forecaster there has no local network to reason over.
        if station.upper() in neighbors.NEIGHBORS:
            return ToolResult(
                f"no obs in the local region: there is no other METAR-reporting airfield "
                f"within {neighbors.MAX_KM:.0f} km of {station}, so there are no neighbour "
                f"observations to compare against. Use the station's own obs (get_latest_obs "
                f"/ get_trend) and the synoptic picture (get_imagery, get_map) instead.")
        return ToolResult(
            f"(no neighbor stations on file for {station}; the nearest-neighbor roster covers "
            "the archived stations -- regenerate with scripts/build_neighbors.py)")
    by_icao = {row[0]: row for row in roster}
    requested = args.get("stations")
    unknown: list[str] = []
    if requested:
        if isinstance(requested, str):
            requested = [requested]
        rows = []
        for want in requested:
            key = str(want).upper()
            if key in by_icao and key not in {r[0] for r in rows}:
                rows.append(by_icao[key])
            elif key not in by_icao:
                unknown.append(key)
        header = f"Requested airfields near {station}, latest observation each."
    else:
        n = _int_arg(args.get("n"), 5, lo=1, hi=5)
        rows = roster[:n]
        header = f"Nearest {len(rows)} airfields to {station}, latest observation each."
    out = [
        f"{header} Distance/bearing are FROM your station; elev is the neighbor minus your field.",
        "decoded cols: UTC time (ISO) | type | wind | vis | ceiling | present-wx | T/Td(C)",
    ]
    skipped = ""
    if unknown:
        skipped = (f"(not in {station}'s fetchable roster, skipped: {', '.join(unknown)}; "
                   f"fetchable are: {', '.join(by_icao)})")
        out.append(skipped)
    n_with_obs = 0
    for icao, dist, brg, de, _la, _lo in rows:
        head = f"{icao}  {dist:.0f} km {brg}  {de:+d} m"
        latest = store.latest(con, icao, 1)
        if not latest:
            out.append(f"{head}  | (no observation in store within the window)")
            continue
        n_with_obs += 1
        r = latest[0]
        out.append(f"{head}  | {_decoded_line(r)}")
        out.append(f"    {r['raw']}")
    if rows and not n_with_obs:
        # Neighbours exist on the map but none of them reported -- distinct from having no
        # neighbours at all, and the model should not be left inferring it from blank rows.
        # Count CHECKED separately from ON FILE: `rows` is the requested subset when the
        # model named stations, so reporting it as the roster size understates the site --
        # asking about one quiet neighbour would read as "this station has one neighbour".
        return ToolResult(
            f"no obs in the local region: none of the {len(rows)} neighbour airfield(s) "
            f"checked ({', '.join(r[0] for r in rows)}) has an observation in the window, so "
            f"there is no local network to compare against. {station} has {len(roster)} "
            f"neighbour airfield(s) on file in total."
            + (f" {skipped}" if skipped else ""))
    return ToolResult("\n".join(out))


def _dirs(ds: list[str], cap: int = 6) -> str:
    """Compass-direction list for the terrain rose: strongest first (already sorted), capped
    for scannability with a trailing '+N more' when the terrain rises/falls many ways."""
    if not ds:
        return "none"
    shown = " ".join(ds[:cap])
    return shown if len(ds) <= cap else f"{shown} (+{len(ds) - cap} more)"


def _map_radius_mi(neigh: list) -> float:
    """Map crop radius (mi): 50 by default, widened so the farthest fetchable neighbor still
    lands on the map (sparse networks -- e.g. PABI -- put neighbors 60-85 mi out)."""
    farthest_km = max((row[1] for row in neigh), default=0.0)
    return max(50.0, round(farthest_km / 1.60934 * 1.15) + 0.0)


def _fmt_terrain(icao: str, p, neigh: list, n_context: int) -> str:
    """Text 'terrain rose' + the fetchable-neighbor index -- the scannable companion to the
    relief map (which plots the same neighbors as blue dots, context sites as violet dots)."""
    reach = max(p.ranges_km)
    lines = [
        f"{icao} terrain (static geography; not a forecast):",
        f"  elevation {p.center_elev_m:.0f} m | relief {p.relief_m:.0f} m within "
        f"{reach:.0f} km | landform: {p.landform}",
        f"  upslope (terrain rises toward): {_dirs(p.upslope)}",
        f"  downslope (terrain falls toward): {_dirs(p.downslope)}",
    ]
    if p.max_rise:
        b, d, r = p.max_rise
        lines.append(f"  steepest rise: +{d:.0f} m to the {b} within {r:.0f} km")
    if p.coast:
        lines.append(f"  nearest coast: {p.coast[0]:.0f} km to the {p.coast[1]}")
    else:
        lines.append("  nearest coast: none within 150 km (inland; inland lakes not detected)")
    if neigh:
        lines.append("  nearby airfields WITH observations (blue dots on map; distance/bearing "
                     "FROM you, elev delta). Pick from these and call get_nearby_obs:")
        for ic, dist, brg, de, _la, _lo in neigh:
            lines.append(f"    {ic}  {dist:.0f} km {brg}  {de:+d} m")
        if n_context:
            lines.append(f"  (+{n_context} more airfields drawn in violet for orientation only -- "
                         "no observations available for those)")
    lines.append("  shaded-relief map (north up; station marked; range rings labeled in mi) "
                 "follows.")
    return "\n".join(lines)


def _get_terrain(args: dict) -> ToolResult:
    """Static terrain + coastline around a station: text rose + relief map with nearby airfields
    plotted (network fetch for elevation, no DB). An unknown ICAO or a fetch failure is
    feedback, not a crash."""
    icao = str(args.get("station") or "").upper()
    if not icao:
        return ToolResult('error: get_terrain needs a "station" ICAO id, e.g. "station": "KVBG"')
    try:
        lat, lon = awc.station_latlon(icao)
    except Exception as e:  # noqa: BLE001 -- unknown id becomes feedback, not a crash
        return ToolResult(f"error: could not resolve a location for {icao} "
                          f"({type(e).__name__}: {e})")
    neigh = neighbors.neighbors_of(icao)
    context = neighbors.area_of(icao)
    markers = [(ic, la, lo) for ic, _d, _b, _e, la, lo in neigh]
    try:
        p = terrain.sample(lat, lon)
        png = terrain.relief_map(lat, lon, markers=markers, context=context,
                                 radius_mi=_map_radius_mi(neigh))
    except Exception as e:  # noqa: BLE001 -- a fetch/render failure becomes feedback
        return ToolResult(f"error: could not build terrain for {icao} ({type(e).__name__}: {e})")
    return ToolResult(_fmt_terrain(icao, p, neigh, len(context)), images=[png])


# ---------------------------------------------------------------------------
# GRIBStream model-data tools. These read the model_data ARCHIVE (populated by
# modeldata.prefetch, network happens OUT of the agent loop) on the read-only conn, so
# they are dispatched in the DB-connected branch of run_tool -- no network here. The
# formatters are lifted from scripts/gribstream_full_demo.py (the blessed product shapes).
# ---------------------------------------------------------------------------

def _k2c(k):
    return None if k is None else k - 273.15


def _ms2kt(ms):
    return None if ms is None else ms * 1.94384


def _inhg_md(hpa):
    """Mean sea-level pressure rendered in inches of mercury, so the model never has to do
    the hPa->inHg arithmetic by hand (it did it inconsistently for the SAME value in round 1).
    NOT the altimeter setting: QNH is reduced from STATION pressure by the standard
    atmosphere, MSLP by the actual temperature profile, so the two diverge as field
    elevation rises. A conversion aid and a trend, not a QNH value to copy."""
    return "--" if hpa is None else f"{hpa / 33.8638866667:.2f}"


def _vis_sm_md(m):
    if m is None:
        return "--"
    sm = m / 1609.34
    return "P6" if sm >= 6 else f"{sm:.1f}"


def _ceil_ft_md(m):
    if m is None or m > 15000 or m < 0:   # fill / no ceiling
        return "none"
    return f"{round(m * 3.28084 / 100) * 100:d}"


def _wind_cell_md(vm: dict, model: str) -> str:
    """Wind cell for a pivoted variable map: NBM stores speed/dir; GFS/HRRR store u/v."""
    if model == "nbm":
        spd, d = vm.get("wind"), vm.get("wdir")
        if spd is None or d is None:
            return "--"
        return f"{int(round(d)) % 360:03d}/{round(_ms2kt(spd)):02d}"
    u, v = vm.get("u10"), vm.get("v10")
    if u is None or v is None:
        return "--"
    dd, ss = _uv_to_dirspd(u, v)
    return f"{dd:03d}/{ss:02d}"


def _pivot_series(rows: list[dict]) -> list[tuple]:
    """Tall model_data rows -> [(valid_time, run, {alias: value})] ordered by valid_time,
    keeping the LATEST run's value for each (valid_time, variable)."""
    cells: dict = {}   # valid_time -> {var: (run, value)}
    for r in rows:
        vt, var, run, val = r["valid_time"], r["variable"], r["run"], r["value"]
        c = cells.setdefault(vt, {})
        if var not in c or (run is not None and c[var][0] is not None and run > c[var][0]):
            c[var] = (run, val)
    out = []
    for vt in sorted(cells):
        varmap = {var: rv[1] for var, rv in cells[vt].items()}
        runs = {rv[0] for rv in cells[vt].values() if rv[0] is not None}
        out.append((vt, max(runs) if runs else None, varmap))
    return out


_WIDE_START = datetime(1970, 1, 1)
_WIDE_END = datetime(2100, 1, 1)


def _resolve_md_location(con, station: str, location: str | None) -> tuple | None:
    """Map a requested location name to (lat, lon, loc_id) from the archive. Defaults to the
    station itself. Returns None if the name isn't pre-fetched (caller lists what is)."""
    want = str(location or station).upper()
    for lc in store.model_data_locations(con):
        if (lc["loc_id"] or "").upper() == want:
            return (lc["lat"], lc["lon"], lc["loc_id"])
    return None


def _md_locations_hint(con) -> str:
    locs = store.model_data_locations(con)
    if not locs:
        return ("(no model data has been pre-fetched into this database; run "
                "scripts/prefetch_model_data.py --station <ICAO>)")
    return "available pre-fetched locations: " + ", ".join(lc["loc_id"] for lc in locs if lc["loc_id"])


def _fmt_model_state(con, station: str, loc: tuple, models: list[str], hours: int | None) -> str:
    lat, lon, loc_id = loc
    blocks: list[str] = []
    peaks: dict = {}
    for model in models:
        rows = store.model_data_series(con, model, lat, lon, start=_WIDE_START, end=_WIDE_END)
        series = _pivot_series(rows)
        if not series:
            continue
        if hours is not None:
            cutoff = series[0][0] + timedelta(hours=hours)
            series = [s for s in series if s[0] <= cutoff]
        run = next((r for _, r, _ in series if r), None)
        lines = [
            f"{model.upper()} surface forecast for {loc_id} -- run "
            f"{run:%Y-%m-%dT%HZ}" if run else f"{model.upper()} surface forecast for {loc_id}",
            f"{'Valid (Z)':<15}{'T C':>5}{'Td C':>6}{'Wind':>8}{'Gst':>5}"
            f"{'MSLP':>7}{'inHg':>7}{'Cld%':>6}{'Vis':>6}{'Ceil ft':>9}",
        ]
        gusts = []
        for vt, _r, vm in series:
            gk = _ms2kt(vm.get("gust"))
            if gk is not None:
                gusts.append(gk)
            t, td, mslp, cld = _k2c(vm.get("t2m")), _k2c(vm.get("td2m")), vm.get("mslp"), vm.get("tcdc")
            lines.append(
                f"{vt:%Y-%m-%dT%HZ}"
                f"{('%5.0f' % t) if t is not None else '   --'}"
                f"{('%6.0f' % td) if td is not None else '    --'}"
                f"{_wind_cell_md(vm, model):>8}"
                f"{('%5.0f' % gk) if gk is not None else '   --'}"
                f"{('%7.0f' % (mslp / 100)) if mslp is not None else '     --'}"
                f"{_inhg_md(None if mslp is None else mslp / 100):>7}"
                f"{('%6.0f' % cld) if cld is not None else '    --'}"
                f"{_vis_sm_md(vm.get('vis')):>6}"
                f"{_ceil_ft_md(vm.get('ceil')):>9}"
            )
        blocks.append("\n".join(lines))
        peaks[model] = max(gusts) if gusts else None
    if not blocks:
        return (f"(no model data pre-fetched for {loc_id}). {_md_locations_hint(con)}")
    synopsis = "  ".join(f"{m.upper()} peak gust {v:.0f}kt" if v else f"{m.upper()} gust --"
                         for m, v in peaks.items())
    return ("\n\n".join(blocks) + f"\n\nCROSS-MODEL: {synopsis}"
            + "\nMSLP is shown in hPa and inHg (same value, converted for you -- do not "
              "re-derive it). It is sea-level pressure: use it for the QNH TREND, not as "
              "the QNH value at an elevated field.")


def _get_model_state(con, station: str, args: dict) -> ToolResult:
    loc = _resolve_md_location(con, station, args.get("location"))
    if loc is None:
        return ToolResult(f"error: {str(args.get('location') or station).upper()} is not a "
                          f"pre-fetched model-data location. {_md_locations_hint(con)}")
    model = args.get("model")
    models = [str(model).lower()] if model else list(modeldata.MODELS)
    hours = args.get("hours")
    hours = _int_arg(hours, hours, lo=1, hi=48) if hours is not None else None
    return ToolResult(_fmt_model_state(con, station, loc, models, hours))


_ICE_LEVELS = ("650 mb", "600 mb", "550 mb", "500 mb", "450 mb", "400 mb")
_VVEL_LEVELS = ("700 mb", "500 mb", "300 mb")


def _pick_valid_time(series: list[tuple], want) -> tuple | None:
    """Choose the series entry nearest a requested valid time (or the first if none asked)."""
    if not series:
        return None
    if want is None:
        return series[0]
    return min(series, key=lambda s: abs((s[0] - want).total_seconds()))


def _fmt_hazard_scan(con, station: str, loc: tuple, want) -> str:
    lat, lon, loc_id = loc
    # read each model's hazard vars, pivot, pick a shared-ish valid time from GFS first
    piv = {}
    for model in ("gfs", "hrrr"):
        rows = store.model_data_series(con, model, lat, lon, start=_WIDE_START, end=_WIDE_END)
        piv[model] = _pivot_series(rows)
    base = piv["gfs"] or piv["hrrr"]
    # The series carries the surface 6h back-tail too, whose early entries hold no pressure-
    # level vars. Restrict to HAZARD-BEARING entries (cape/t650 present) BEFORE choosing, for
    # a requested valid time as well as for the default -- an entry with no levels renders an
    # empty panel however it was picked, so the nearest level-bearing time is the honest answer.
    ref = _pick_valid_time(
        [e for e in base if e[2].get("cape") is not None or e[2].get("t650") is not None], want)
    if ref is None:
        # The level bundle is SITE-COLUMN ONLY (modeldata.hazard_coords, config B), but this
        # tool forwards `location`, so a neighbour or grid point resolves on its SURFACE row
        # and has no levels at ANY time. That is why this tests for a level-bearing ENTRY and
        # not for rows: `base` is non-empty at such a point, so a rows test never fires here
        # and the scan renders a confident header over a blank panel. Name that limit instead
        # of listing locations that would mostly repeat it.
        if loc_id.upper() != station.upper():
            return (f"(no pressure-level data for {loc_id}: the model-data archive pulls "
                    f"pressure levels for the SITE COLUMN only, so neighbour and grid points "
                    f"carry surface fields but no levels. Re-run this scan for "
                    f"{station.upper()} itself for the hazard picture.)")
        return (f"(no pressure-level hazard data pre-fetched for {loc_id} -- the pressure-level "
                f"bundle was not pulled for this cycle.)")
    valid = ref[0]
    out = [f"Hazard scan for {loc_id}, valid {valid:%Y-%m-%dT%HZ} -- conditions diagnosed from "
           "GFS + HRRR (no native icing/turbulence field; we confirm the ENVIRONMENT across "
           "models). Reason over the evidence; the flags are a rule, not a verdict.", ""]

    # ICING: supercooled water (T in [-16,0] C, RH>=70%; GFS CLMR>0 confirms cloud liquid)
    out.append("ICING (T in -16..0 C with RH>=70%; GFS CLMR>0 confirms supercooled liquid):")
    ice: dict = {}
    for model in ("gfs", "hrrr"):
        entry = _pick_valid_time(piv[model], valid)
        if entry is None:
            out.append(f"  {model.upper()}: no data at valid time")
            continue
        _vt, run, vm = entry
        out.append(f"  {model.upper()} (run {run:%Y-%m-%dT%HZ}):" if run else f"  {model.upper()}:")
        for lv in _ICE_LEVELS:
            p = lv[:3]
            t, rh = _k2c(vm.get(f"t{p}")), vm.get(f"rh{p}")
            if t is None or rh is None:
                continue
            clw = vm.get(f"clw{p}")
            flag = (-16.0 <= t <= 0.0) and rh >= 70.0
            clw_s = f" CLW={clw * 1000:.2f}g/kg" if clw is not None else ""
            ice.setdefault(lv, {})[model] = flag
            out.append(f"    {lv:<7} T={t:>5.1f}C RH={rh:>3.0f}%{clw_s:<16} "
                       f"{'ICING' if flag else '-'}")
    if ice:
        out.append("  agreement: " + "; ".join(
            f"{lv} " + ("BOTH icing" if set(v.values()) == {True}
                        else "no icing" if set(v.values()) == {False} else f"DISAGREE {v}")
            for lv, v in ice.items() if v))

    # TURBULENCE: convective (CAPE + ascent) and shear-driven (deep-layer bulk shear)
    out += ["", "TURBULENCE (convective: CAPE + ascent; mechanical/CAT: 850-300mb bulk shear):"]
    summ: dict = {}
    for model in ("gfs", "hrrr"):
        entry = _pick_valid_time(piv[model], valid)
        if entry is None:
            continue
        _vt, run, vm = entry
        cape, cin = vm.get("cape"), vm.get("cin")
        w = {lv[:3]: vm.get(f"w{lv[:3]}") for lv in _VVEL_LEVELS}
        max_up = min((x for x in w.values() if x is not None), default=None)  # omega<0 = up
        u8, v8, u3, v3 = vm.get("u850"), vm.get("v850"), vm.get("u300"), vm.get("v300")
        deep = (_ms2kt(math.hypot(u3 - u8, v3 - v8)) if None not in (u8, v8, u3, v3) else None)
        summ[model] = (cape, deep)
        parts = [f"CAPE={cape:.0f}J/kg" if cape is not None else "CAPE=--",
                 f"CIN={cin:.0f}" if cin is not None else "CIN=--",
                 f"max ascent(omega)={max_up:.1f}Pa/s" if max_up is not None else "omega=--",
                 f"850-300mb shear={deep:.0f}kt" if deep is not None else "shear=--"]
        if model == "gfs" and vm.get("hlcy") is not None:
            parts.append(f"SRH(0-3km)={vm['hlcy']:.0f}m2/s2")
        out.append(f"  {model.upper()} (run {run:%Y-%m-%dT%HZ}): " + ", ".join(parts)
                   if run else f"  {model.upper()}: " + ", ".join(parts))
    if len(summ) == 2:
        cg, ch = summ["gfs"][0], summ["hrrr"][0]
        sg, sh = summ["gfs"][1], summ["hrrr"][1]
        conv = ("BOTH show convective potential" if (cg or 0) > 500 and (ch or 0) > 500
                else "single-model convective signal" if (cg or 0) > 500 or (ch or 0) > 500
                else "neither model convective")
        shr = ("deep shear >40kt in both (organized/CAT)" if (sg or 0) > 40 and (sh or 0) > 40
               else "modest shear")
        out.append(f"  agreement: {conv}; {shr}")
    return "\n".join(out)


def _get_hazard_scan(con, station: str, args: dict) -> ToolResult:
    loc = _resolve_md_location(con, station, args.get("location"))
    if loc is None:
        return ToolResult(f"error: {str(args.get('location') or station).upper()} is not a "
                          f"pre-fetched model-data location. {_md_locations_hint(con)}")
    want = None
    if args.get("valid_time"):
        try:
            want = datetime.strptime(str(args["valid_time"]).replace("Z", "")[:16], "%Y-%m-%dT%H:%M")
        except ValueError:
            return ToolResult('error: valid_time must be ISO like "2026-07-17T21:00Z"')
    return ToolResult(_fmt_hazard_scan(con, station, loc, want))


# --- GEFS ensemble probability product --------------------------------------------------
_WIND_THRESHOLDS = (15, 25, 35, 45)   # kt exceedance rungs for wind speed AND gust
_ENS_ALIASES = ("t2m", "td2m", "u10", "v10", "gust", "vis", "ceil")


def _pivot_by_member(rows: list[dict]) -> dict:
    """Tall GEFS rows -> {valid_time: {member: {alias: value}}} for the LATEST run only.
    Each member is a full forecast; the reader turns the spread across members at one hour
    into a probability."""
    latest: dict = {}
    for r in rows:
        vt = r["valid_time"]
        run = r.get("run")
        if run is not None and (vt not in latest or run > latest[vt]):
            latest[vt] = run
    out: dict = {}
    for r in rows:
        vt = r["valid_time"]
        if r.get("run") != latest.get(vt):
            continue
        out.setdefault(vt, {}).setdefault(r.get("member", 0), {})[r["variable"]] = r["value"]
    return out


def _pct(vals: list[float], q: float) -> float | None:
    """Linear-interpolated percentile of a value list (q in 0..100)."""
    xs = sorted(v for v in vals if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q / 100.0
    lo = int(k)
    frac = k - lo
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def _member_ceiling_cat(m, profile) -> str | None:
    """One member's ceiling (meters, GFS-style HGT@cloud ceiling) -> TAFVER band id."""
    if m is None or m > 15000:               # fill / no ceiling -> unlimited (top band)
        return tafstate.tafver_ceiling_category(None, tafstate.CEIL_KNOWN_UNLIMITED, profile)
    return tafstate.tafver_ceiling_category(m * 3.28084, tafstate.CEIL_KNOWN_NUMERIC, profile)


def _member_vis_cat(m, profile) -> str | None:
    """One member's visibility (meters) -> TAFVER band id (statute-mile ladder)."""
    if m is None:
        return None
    sm = m / 1609.34
    if sm >= 6.0:                            # >=6 SM is reported unlimited (P6SM)
        return tafstate.tafver_visibility_category(None, None, tafstate.VIS_KNOWN_UNLIMITED, profile)
    return tafstate.tafver_visibility_category(sm, None, tafstate.VIS_KNOWN_NUMERIC, profile)


def _cat_prob_row(members: dict, classify, band_ids: list[str]) -> tuple[dict, int]:
    """Fraction of members whose value falls in each band. Returns (per-band fraction, n
    members that classified) -- a member with a missing/unknown value is left out of n."""
    counts = {b: 0 for b in band_ids}
    n = 0
    for vm in members.values():
        cid = classify(vm)
        if cid is None:
            continue
        n += 1
        counts[cid] = counts.get(cid, 0) + 1
    return ({b: (counts[b] / n if n else None) for b in band_ids}, n)


def _exceed_prob(members: dict, value_fn, thresholds) -> tuple[dict, int]:
    """Fraction of members whose derived value is >= each threshold."""
    vals = [value_fn(vm) for vm in members.values()]
    vals = [v for v in vals if v is not None]
    n = len(vals)
    return ({t: (sum(1 for v in vals if v >= t) / n if n else None) for t in thresholds}, n)


def _pp(p) -> str:
    """A probability 0..1 as a 2-wide percent cell; blank when undefined."""
    return " --" if p is None else f"{round(p * 100):3d}"


def _fmt_ensemble_prob(con, station: str) -> str:
    lat, lon, _ = _resolve_md_location(con, station, None) or (None, None, None)
    if lat is None:
        return f"error: {station} has no pre-fetched GEFS ensemble. {_md_locations_hint(con)}"
    rows = store.model_data_series(con, modeldata.GEFS_MODEL, lat, lon, start=_WIDE_START,
                                   end=_WIDE_END, variables=list(_ENS_ALIASES))
    by_vt = _pivot_by_member(rows)
    if not by_vt:
        return (f"error: no GEFS ensemble archived for {station}. A deterministic model may be "
                f"present, but the ensemble product needs a prefetch_ensemble pull.")
    profile = tafstate.default_profile(station)
    cig_bands = [b.id for b in profile.tafver_ceiling_bands]
    vis_bands = [b.id for b in profile.tafver_vis_bands]
    cig_lbl = {"A": "<2", "B": "2-7", "C": "7-10", "D": "10-20", "E": ">=20"}   # hundreds ft
    vis_lbl = {"A": "<0.5", "B": ".5-2", "C": "2-3", "D": "2-3", "E": ">=3"}    # statute miles
    hours = sorted(by_vt)
    n_members = max((len(m) for m in by_vt.values()), default=0)
    out = [
        f"GEFS ensemble probabilities for {station} -- {n_members} members, run "
        f"{max((r.get('run') for r in rows if r.get('run')), default='?')}. Each cell is the "
        "PERCENT of members meeting the condition at that hour; read a column down to see when "
        "a risk grows.",
        "CEILING and VISIBILITY use the TAFVER category bands (ceiling in hundreds of feet, "
        "vis in statute miles); the percentages across a row sum to ~100 (every member lands "
        "in one band). WIND and GUST are EXCEEDANCE: percent of members at or above each knot "
        "threshold, so they do not sum to 100.",
        "T/Td show the ensemble spread as p10/p50/p90 (C): the median with a 10th-90th "
        "percentile band. A wide band = low confidence.",
        "",
        "Ceiling category (hundreds ft): " + "  ".join(f"{b}={cig_lbl[b]}" for b in cig_bands),
        "Visibility category (SM):       " + "  ".join(f"{b}={vis_lbl[b]}" for b in vis_bands),
        "",
    ]
    # Ceiling block
    out.append("CEILING -- % of members in each category")
    out.append("  Valid" + "".join(f"{b:>6}" for b in cig_bands) + "    n")
    for vt in hours:
        probs, n = _cat_prob_row(by_vt[vt], lambda vm: _member_ceiling_cat(vm.get("ceil"), profile),
                                 cig_bands)
        out.append(f"  {vt:%d/%HZ}" + "".join(f"{_pp(probs[b]):>6}" for b in cig_bands) + f"{n:>5}")
    out.append("")
    # Visibility block
    out.append("VISIBILITY -- % of members in each category")
    out.append("  Valid" + "".join(f"{b:>6}" for b in vis_bands) + "    n")
    for vt in hours:
        probs, n = _cat_prob_row(by_vt[vt], lambda vm: _member_vis_cat(vm.get("vis"), profile),
                                 vis_bands)
        out.append(f"  {vt:%d/%HZ}" + "".join(f"{_pp(probs[b]):>6}" for b in vis_bands) + f"{n:>5}")
    out.append("")
    # Wind + gust exceedance
    for label, fn in (("WIND SPEED", lambda vm: _ens_wind_kt(vm)),
                      ("WIND GUST", lambda vm: _ms2kt(vm.get("gust")))):
        out.append(f"{label} -- % of members >= threshold (kt)")
        out.append("  Valid" + "".join(f"{f'>={t}':>6}" for t in _WIND_THRESHOLDS) + "    n")
        for vt in hours:
            probs, n = _exceed_prob(by_vt[vt], fn, _WIND_THRESHOLDS)
            out.append(f"  {vt:%d/%HZ}"
                       + "".join(f"{_pp(probs[t]):>6}" for t in _WIND_THRESHOLDS) + f"{n:>5}")
        out.append("")
    # Temperature / dewpoint percentiles
    out.append("TEMPERATURE / DEWPOINT -- ensemble p10/p50/p90 (C)")
    out.append(f"  {'Valid':<8}{'T p10/p50/p90':>18}{'Td p10/p50/p90':>20}")
    for vt in hours:
        ts = [_k2c(vm.get("t2m")) for vm in by_vt[vt].values()]
        ds = [_k2c(vm.get("td2m")) for vm in by_vt[vt].values()]
        out.append(f"  {vt:%d/%HZ}   "
                   f"{_tri(ts):>15}{_tri(ds):>20}")
    return "\n".join(out)


def _ens_wind_kt(vm: dict):
    u, v = vm.get("u10"), vm.get("v10")
    if u is None or v is None:
        return None
    return _ms2kt(math.hypot(u, v))


def _tri(vals: list) -> str:
    p10, p50, p90 = _pct(vals, 10), _pct(vals, 50), _pct(vals, 90)
    if p50 is None:
        return "--"
    return f"{p10:.0f}/{p50:.0f}/{p90:.0f}"


def _get_ensemble_prob(con, station: str, args: dict) -> ToolResult:
    return ToolResult(_fmt_ensemble_prob(con, station))


_VER_ALIASES = ("t2m", "td2m", "u10", "v10", "wind", "wdir", "gust", "mslp")

# How many model runs to show per model, newest first. Several runs is the POINT of this
# view -- seeing the same hour forecast by successive runs is how the reader learns that
# the fresher run is closer, without being told so in the prompt.
_VER_MAX_RUNS = 3


def _pivot_by_run(rows: list[dict]) -> dict:
    """Tall model_data rows -> {run: {valid_time: {alias: value}}}.

    Unlike _pivot_series (which collapses to the LATEST run per valid time, right for a
    'what does guidance say now' view) this keeps every run separate, so the same valid
    hour can be shown as forecast by successive runs."""
    out: dict = {}
    for r in rows:
        if r["run"] is None:
            continue
        out.setdefault(r["run"], {}).setdefault(r["valid_time"], {})[r["variable"]] = r["value"]
    return out


def _obs_by_hour(con, station: str, lo: datetime, hi: datetime) -> dict:
    """Observations keyed to the hour they describe. A :55 report describes the top of the
    NEXT hour, so it is snapped forward. When several reports land in one hour, the routine
    METAR wins and a SPECI is used only if there is no routine one -- SPECIs are issued
    BECAUSE conditions changed, so letting an arbitrary one win would make the comparison
    depend on which report happened to be read last."""
    best: dict = {}
    for o in store.window(con, station, lo - timedelta(hours=1), hi + timedelta(hours=1)):
        key = (o["obs_time"] + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
        cur = best.get(key)
        if cur is None or (cur.get("report_type") != "METAR" and o.get("report_type") == "METAR"):
            best[key] = o
    return best


def _ms2kt(v):
    return None if v is None else v * 1.94384


def _pa2inhg(v):
    return None if v is None else v / 3386.389


def _fcst_wind(vm: dict) -> tuple:
    """(direction, speed kt) from whichever wind form the model carries -- GFS/HRRR give
    u/v components, NBM gives speed + direction directly."""
    if vm.get("wind") is not None or vm.get("wdir") is not None:
        return vm.get("wdir"), _ms2kt(vm.get("wind"))
    u, v = vm.get("u10"), vm.get("v10")
    if u is None or v is None:
        return None, None
    return math.degrees(math.atan2(-u, -v)) % 360, _ms2kt(math.hypot(u, v))


def _deg_err(f, o):
    return None if f is None or o is None else (f - o + 180) % 360 - 180


def _fo(f, o, n=0):
    """'forecast/observed' as one cell, so the reader sees the values, not just the error."""
    fs = "--" if f is None else f"{f:.{n}f}"
    os_ = "--" if o is None else f"{o:.{n}f}"
    return f"{fs}/{os_}"


def _es(v, n=1):
    return "--" if v is None else f"{v:+.{n}f}"


# One block per FIELD, rows = valid hour, columns = model run. The earlier layout gave each
# run its own block, which reads fine but cannot answer the question the tool exists to
# answer -- "is the fresher run closer?" needs the SAME hour on ONE line across runs.
# (label, observed getter, forecast getter, value decimals, error decimals, circular)
_VER_FIELDS = (
    ("TEMPERATURE (C)", lambda o: o.get("temp_c"),
     lambda vm, d, s: _k2c(vm.get("t2m")), 0, 1, False),
    ("DEWPOINT (C)", lambda o: o.get("dewpoint_c"),
     lambda vm, d, s: _k2c(vm.get("td2m")), 0, 1, False),
    ("ALTIMETER / MSLP (inHg)", lambda o: o.get("altimeter_inhg"),
     lambda vm, d, s: _pa2inhg(vm.get("mslp")), 2, 2, False),
    ("WIND DIRECTION (deg)", lambda o: o.get("wind_dir_deg"),
     lambda vm, d, s: d, 0, 0, True),
    ("WIND SPEED (kt)", lambda o: o.get("wind_speed"),
     lambda vm, d, s: s, 0, 1, False),
    # A report with no gust group means no gusts occurred -- a real value (0), not missing
    # data -- so a model forecasting gusts that never happened shows its error, not a blank.
    ("WIND GUST (kt)", lambda o: o.get("wind_gust") or 0,
     lambda vm, d, s: _ms2kt(vm.get("gust")), 0, 1, False),
)


def _fmt_field_block(label: str, hours: list, runs: list, by_run: dict, obs: dict,
                     obs_get, fcst_get, vdp: int, edp: int, circular: bool) -> str:
    """One field's hour x run table plus the two summary lines per run."""
    w = 14
    head = [f"{label} -- forecast (error vs observed)",
            f"  {'Valid':<7}{'obs':>7}  " + "".join(f"{f'{r:%HZ} run':>{w}}" for r in runs)]
    acc: dict = {r: [] for r in runs}
    body = []
    for vt in hours:
        o = obs_get(obs[vt])
        cells = []
        for r in runs:
            vm = by_run[r].get(vt)
            if vm is None:
                cells.append(f"{'--':>{w}}")
                continue
            d, s = _fcst_wind(vm)
            f = fcst_get(vm, d, s)
            if f is None:
                cells.append(f"{'--':>{w}}")
                continue
            err = _deg_err(f, o) if circular else (None if o is None else f - o)
            if err is not None:
                acc[r].append(err)
            cells.append(f"{f'{f:.{vdp}f} ({err:+.{edp}f})' if err is not None else f'{f:.{vdp}f}':>{w}}")
        stamp = f"{vt:%d/%H}Z"
        body.append(f"  {stamp:<7}{'--' if o is None else f'{o:.{vdp}f}':>7}  " + "".join(cells))
    means = "".join(
        f"{('--' if not acc[r] else f'{sum(acc[r]) / len(acc[r]):+.{edp}f}'):>{w}}" for r in runs)
    typ = "".join(
        f"{('--' if not acc[r] else f'{sum(abs(v) for v in acc[r]) / len(acc[r]):.{edp}f}'):>{w}}"
        for r in runs)
    counts = "".join(f"{f'n={len(acc[r])}':>{w}}" for r in runs)
    return "\n".join(head + body + [
        f"  {'mean err':<7}{'':>7}  " + means,
        f"  {'typical':<7}{'':>7}  " + typ,
        f"  {'hours':<7}{'':>7}  " + counts,
    ])


def _fmt_model_verification_by_hour(con, station: str, models: list[str]) -> str:
    """Hour x run verification: every field, every hour, one column per model run."""
    lat, lon, _ = _resolve_md_location(con, station, None) or (None, None, None)
    if lat is None:
        return f"error: {station} is not a pre-fetched model-data location. {_md_locations_hint(con)}"
    out = [
        f"Model-vs-obs verification for {station}. Each block is one field: rows are valid "
        "hours, columns are MODEL RUNS. A cell is the run's forecast with its error vs the "
        "observed report in brackets; error is forecast minus observed.",
        "Read ACROSS a row to see whether the fresher run was closer for that hour. A run "
        "cannot forecast hours before it started, so older runs fill more of the table.",
        "Two summary lines per run: 'mean err' is the average SIGNED error -- a bias you can "
        "correct by subtracting it. 'typical' is the average error SIZE, which is how far off "
        "to expect any single hour to be. When mean err is small but typical is large the "
        "errors are cancelling, not absent: there is no bias to correct and the field is "
        "simply unreliable. Judge reliability by 'typical', correct bias by 'mean err'.",
        "Negative T/Td = model too cold/dry. QNH compares model MSLP against the observed "
        "altimeter setting: close, but not the same quantity, so read a small steady offset "
        "with care.",
        "",
    ]
    matched_any = False
    for model in models:
        rows = store.model_data_series(con, model, lat, lon, start=_WIDE_START, end=_WIDE_END,
                                       variables=list(_VER_ALIASES))
        by_run = _pivot_by_run(rows)
        if not by_run:
            continue
        runs = sorted(by_run, reverse=True)[:_VER_MAX_RUNS]
        all_vt = [vt for r in runs for vt in by_run[r]]
        if not all_vt:
            continue
        obs = _obs_by_hour(con, station, min(all_vt), max(all_vt))
        hours = sorted({vt for r in runs for vt in by_run[r] if vt in obs})
        if not hours:
            continue
        matched_any = True
        out.append(f"=== {model.upper()} -- runs "
                   + ", ".join(f"{r:%Y-%m-%dT%HZ}" for r in runs)
                   + f" -- {len(hours)} verified hours ===")
        out.append("")
        for label, obs_get, fcst_get, vdp, edp, circ in _VER_FIELDS:
            out.append(_fmt_field_block(label, hours, runs, by_run, obs,
                                        obs_get, fcst_get, vdp, edp, circ))
            out.append("")
    if not matched_any:
        out.append("(no observed reports overlap the archived forecast valid times yet -- "
                   "verification needs obs at the pre-issue forecast hours in the store)")
    return "\n".join(out)


def _fmt_model_verification(con, station: str, models: list[str]) -> str:
    lat, lon, _ = _resolve_md_location(con, station, None) or (None, None, None)
    if lat is None:
        return f"error: {station} is not a pre-fetched model-data location. {_md_locations_hint(con)}"
    # obs truth from the DB (leakage-safe: the per-run DB is cut off at issue time).
    out = [f"Model-vs-obs verification for {station} -- archived forecast (from runs <= issue) "
           "vs observed reports at the matching hours. Each cell is forecast/observed; err is "
           "forecast minus observed.",
           "Negative T/Td = model too cold/dry. Positive QNH = model pressure too high. "
           "dir/spd/gust are the 10 m wind in degrees and knots.",
           "A run cannot forecast hours before it started, so later runs begin later. Compare "
           "the SAME hour across runs to see whether the fresher run is closer.",
           "QNH compares model mean-sea-level pressure against the observed altimeter setting: "
           "close, but not the same quantity, so read a small steady offset with care.",
           ""]
    matched_any = False
    for model in models:
        rows = store.model_data_series(con, model, lat, lon, start=_WIDE_START, end=_WIDE_END,
                                       variables=list(_VER_ALIASES))
        by_run = _pivot_by_run(rows)
        if not by_run:
            continue
        all_vt = [vt for r in by_run.values() for vt in r]
        obs = _obs_by_hour(con, station, min(all_vt), max(all_vt))
        for run in sorted(by_run, reverse=True)[:_VER_MAX_RUNS]:
            hours = sorted(vt for vt in by_run[run] if vt in obs)
            if not hours:
                continue
            block = [f"{model.upper()} run {run:%Y-%m-%dT%HZ}:",
                     f"  {'Valid':<7}{'T f/o':>8}{'err':>6}{'Td f/o':>8}{'err':>6}"
                     f"{'QNH f/o':>14}{'err':>7}{'dir f/o':>10}{'err':>6}"
                     f"{'spd f/o':>9}{'err':>6}{'gust f/o':>10}{'err':>6}"]
            acc: dict = {}
            for vt in hours:
                vm, o = by_run[run][vt], obs[vt]
                tf, tdf = _k2c(vm.get("t2m")), _k2c(vm.get("td2m"))
                qf = _pa2inhg(vm.get("mslp"))
                df, sf = _fcst_wind(vm)
                gf = _ms2kt(vm.get("gust"))
                to, tdo, qo = o.get("temp_c"), o.get("dewpoint_c"), o.get("altimeter_inhg")
                do, so = o.get("wind_dir_deg"), o.get("wind_speed")
                # A report with no gust group means no gusts occurred, which is a real
                # value (0), not missing data -- otherwise a model forecasting gusts that
                # never happened would show a blank instead of its error.
                go = o.get("wind_gust") or 0
                sub = lambda f, ob: None if f is None or ob is None else f - ob  # noqa: E731
                errs = {"T": sub(tf, to), "Td": sub(tdf, tdo), "QNH": sub(qf, qo),
                        "dir": _deg_err(df, do), "spd": sub(sf, so), "gust": sub(gf, go)}
                for k, v in errs.items():
                    if v is not None:
                        acc.setdefault(k, []).append(v)
                block.append(
                    f"  {vt:%H}Z{'':<4}{_fo(tf, to):>8}{_es(errs['T']):>6}"
                    f"{_fo(tdf, tdo):>8}{_es(errs['Td']):>6}"
                    f"{_fo(qf, qo, 2):>14}{_es(errs['QNH'], 2):>7}"
                    f"{_fo(df, do):>10}{_es(errs['dir'], 0):>6}"
                    f"{_fo(sf, so):>9}{_es(errs['spd']):>6}"
                    f"{_fo(gf, go):>10}{_es(errs['gust']):>6}")
            matched_any = True
            mean = {k: sum(v) / len(v) for k, v in acc.items()}
            # Direction errors cancel: +40 one hour and -40 the next average to zero while
            # every hour was badly wrong. Report the typical SIZE of the miss beside it.
            typ = (sum(abs(v) for v in acc["dir"]) / len(acc["dir"])) if acc.get("dir") else None
            block.append(
                f"  mean err over {len(hours)} hrs:  T {_es(mean.get('T'))}C   "
                f"Td {_es(mean.get('Td'))}C   QNH {_es(mean.get('QNH'), 2)}inHg   "
                f"dir {_es(mean.get('dir'), 0)}deg (typically "
                f"{'--' if typ is None else f'{typ:.0f}'}deg off)   "
                f"spd {_es(mean.get('spd'))}kt   gust {_es(mean.get('gust'))}kt")
            out.append("\n".join(block))
            out.append("")
    if not matched_any:
        out.append("(no observed reports overlap the archived forecast valid times yet -- "
                   "verification needs obs at the pre-issue forecast hours in the store)")
    return "\n".join(out)


def _get_model_verification(con, station: str, args: dict) -> ToolResult:
    model = args.get("model")
    models = [str(model).lower()] if model else list(modeldata.MODELS)
    return ToolResult(_fmt_model_verification_by_hour(con, station, models))


# Human-readable unit hints for the spatial field tool's common aliases.
_FIELD_UNITS = {"t2m": ("C", _k2c), "td2m": ("C", _k2c), "gust": ("kt", _ms2kt),
                "vis": ("SM", lambda m: None if m is None else m / 1609.34),
                "mslp": ("hPa", lambda p: None if p is None else p / 100)}


def _fmt_nearby_model_data(con, model: str, variable: str, want) -> str:
    # find a stored valid time nearest `want` (or the first) by checking one location
    locs = store.model_data_locations(con)
    if not locs:
        return _md_locations_hint(con)
    ref = None
    for lc in locs:
        vts = store.model_data_valid_times(con, model, lc["lat"], lc["lon"])
        if vts:
            ref = min(vts, key=lambda v: abs((v - want).total_seconds())) if want else vts[0]
            break
    if ref is None:
        return f"(no {model.upper()} data pre-fetched). {_md_locations_hint(con)}"
    field = store.model_data_field(con, model, variable, valid_time=ref)
    if not field:
        return (f"(no {model.upper()} '{variable}' at {ref:%Y-%m-%dT%HZ}; check the alias -- "
                "surface aliases: t2m td2m gust mslp vis ceil tcdc; wind is u10/v10 or wind/wdir)")
    unit, conv = _FIELD_UNITS.get(variable, ("native", lambda x: x))
    out = [f"{model.upper()} '{variable}' ({unit}) across pre-fetched points, valid "
           f"{ref:%Y-%m-%dT%HZ} -- for gradient/advection reasoning (sorted by location id):",
           f"  {'loc':<10}{'lat':>9}{'lon':>10}{'value':>10}"]
    for r in field:
        cv = conv(r["value"])
        vs = "--" if cv is None else (f"{cv:.1f}" if unit != "native" else f"{cv:.3g}")
        out.append(f"  {(r['loc_id'] or ''):<10}{r['lat']:>9.4f}{r['lon']:>10.4f}{vs:>10}")
    return "\n".join(out)


def _get_nearby_model_data(con, station: str, args: dict) -> ToolResult:
    variable = str(args.get("variable") or "").strip()
    if not variable:
        return ToolResult('error: get_nearby_model_data needs a "variable" alias, e.g. '
                          '"variable": "t2m" (surface: t2m td2m gust mslp vis ceil tcdc)')
    model = str(args.get("model") or "gfs").lower()
    if model not in modeldata.MODELS:
        return ToolResult(f"error: unknown model {model!r}; choose from {', '.join(modeldata.MODELS)}")
    want = None
    if args.get("valid_time"):
        try:
            want = datetime.strptime(str(args["valid_time"]).replace("Z", "")[:16], "%Y-%m-%dT%H:%M")
        except ValueError:
            return ToolResult('error: valid_time must be ISO like "2026-07-17T21:00Z"')
    return ToolResult(_fmt_nearby_model_data(con, model, variable, want))


def _stamp_fetched(result: ToolResult) -> ToolResult:
    """Append the UTC fetch time to a network receipt, unless the fetch errored. The
    cycle/valid time of model-run products is already on the receipt; this pins the
    live/analysis products to when the model actually saw them, so the archived context
    is unambiguous after the fact."""
    if result.text and not result.text.startswith("error:"):
        result.text = f"{result.text}\n(fetched {datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ})"
    return result


def run_tool(name: str, args: dict, *, db_path: str | None = None,
             evidence_ids: list[str] | None = None,
             station: str | None = None) -> ToolResult:
    """Execute a model-issued tool call. The read tools run against a READ-ONLY
    connection; the sinks (emit_taf, check_taf, submit_taf_worksheet) and the network
    fetches (get_current_taf, get_sounding, get_map, get_imagery, get_loop, get_terrain)
    need no DB and are handled first. `evidence_ids`
    (the ids the loop has threaded) lets submit_taf_worksheet RESOLVE evidence_refs.
    Returns a ToolResult: text receipt + images/TAF/worksheet."""
    if name == "emit_taf":
        return _emit_taf(args)
    if name == "check_taf":
        return _check_taf(args)
    if name == "submit_taf_worksheet":
        return _submit_worksheet(args, evidence_ids=evidence_ids)
    # Network fetches: no DB, handled before the read-only connect. Each is stamped with
    # its fetch time so a 'now' product (analysis map, satellite, radar, the live TAF) is
    # pinned to the instant the model saw it -- model-run products also cite their cycle.
    if name == "get_current_taf":
        return _stamp_fetched(_get_current_taf(args))
    if name == "get_sounding":
        return _stamp_fetched(_get_sounding(args))
    if name == "get_map":
        return _stamp_fetched(_get_map(args, station))
    if name == "get_imagery":
        return _stamp_fetched(_get_imagery(args))
    if name == "get_loop":
        return _stamp_fetched(_get_loop(args))
    if name == "get_terrain":
        return _stamp_fetched(_get_terrain(args))
    con = (
        store.connect(db_path, read_only=True)
        if db_path
        else store.connect(read_only=True)
    )
    try:
        station = args.get("station")
        if not station:
            return ToolResult('error: this tool needs a "station" ICAO id, e.g. "station": "KBLV"')
        station = str(station).upper()
        if name == "get_previous_taf":
            return _get_previous_taf(con, station)
        if name == "query_obs":
            start, end, err = _resolve_window(con, station, args)
            if err:
                return ToolResult(f"error: {err}")
            rows = store.window(con, station, start, end)
            return ToolResult(
                _window_line(start, end) + "\n" + _fmt(rows, "oldest first"),
                window=(start, end),
            )
        if name == "get_latest_obs":
            n = _int_arg(args.get("n"), 1, lo=1, hi=200)
            rows = store.latest(con, station, n)
            return ToolResult(_fmt(rows, "newest first"))
        if name == "get_trend":
            hours = _int_arg(args.get("hours"), 24, lo=1, hi=48)  # coerce + clamp the look-back
            anchor = store.latest(con, station, 1)
            if not anchor:
                return ToolResult(f"(no observations for {station})")
            end = anchor[0]["obs_time"]
            start = end - timedelta(hours=hours)
            rows = store.window(con, station, start, end)
            if not rows:
                return ToolResult(
                    f"{_window_line(start, end)}\n"
                    f"(no observations for {station} in the last {hours}h)"
                )
            png = charts.meteogram(rows, station=station, hours=hours)
            receipt = (
                f"{_window_line(start, end)}\n"
                f"Meteogram for {station}, last {hours}h ({len(rows)} obs); image follows."
            )
            return ToolResult(receipt, images=[png], window=(start, end))
        if name == "get_nearby_obs":
            return _get_nearby_obs(con, station, args)
        if name == "get_climo":
            return _get_climo(con, args)
        if name == "get_fcst_sounding":
            return _get_fcst_sounding(con, station, args)
        if name == "get_point_forecast":
            return _get_point_forecast(con, station, args)
        if name == "get_model_state":
            return _get_model_state(con, station, args)
        if name == "get_hazard_scan":
            return _get_hazard_scan(con, station, args)
        if name == "get_model_verification":
            return _get_model_verification(con, station, args)
        if name == "get_nearby_model_data":
            return _get_nearby_model_data(con, station, args)
        if name == "get_ensemble_prob":
            return _get_ensemble_prob(con, station, args)
        return ToolResult(f"error: unknown tool {name!r}")
    except Exception as e:  # noqa: BLE001 -- any read-tool failure becomes feedback, not a dead loop
        return ToolResult(f"error: {name} failed ({type(e).__name__}: {e})")
    finally:
        con.close()


# _image_mime stays here: it is a tool-output format helper (tools.py sniffs image
# bytes for get_imagery AND for the ToolResult images the agent loop renders). The
# agent-loop plumbing (final_answer, tool_messages, window_conflict) lives in agent.py.
def _image_mime(data: bytes) -> str:
    """Content type from magic bytes. A meteogram is PNG, but a fetched skew-T can be
    a GIF (SPC) or PNG (Wyoming), and a vision model rejects an image whose data URL
    lies about its type -- so label each image by what it actually is."""
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/png"
