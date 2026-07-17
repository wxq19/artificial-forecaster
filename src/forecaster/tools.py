"""Agent-facing tools.

The model can only emit a tool CALL (structured JSON like
{"name": "query_obs", "station": "KORD", ...}); it never sees SQL or a
connection. This module validates the call and runs it against a READ-ONLY
DuckDB connection, so a hallucinated tool call physically cannot write or delete.
Only read tools are registered here — that's the menu the model is limited to.
Results come back as compact text the VLM can reason over.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from forecaster import (
    awc, charts, fcstsounding, imagery, modeldata, neighbors, soundings, store, tafgen,
    tafparse, terrain, worksheet, wxmaps,
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
            "Fetch an observed upper-air skew-T sounding (radiosonde) as an image to "
            "judge vertical structure: stability/CAPE, inversions, moisture layers, "
            "freezing level, and wind shear with height. Soundings are launched only "
            "at 00Z and 12Z from upper-air sites (NOT every airport); you get the most "
            "recent synoptic run at or before now. `site` is an upper-air station id "
            "(e.g. OUN, MPX), which may differ from the nearest airport's ICAO."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "site": {
                    "type": "string",
                    "description": "Upper-air sounding site id, e.g. OUN or MPX",
                },
                "source": {
                    "type": "string",
                    "enum": ["spc", "wyoming"],
                    "description": "Provider: spc (default, richer analysis) or wyoming",
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
            "Fetch a MODEL FORECAST sounding (skew-T image) for an airport at a chosen "
            "forecast hour -- the PREDICTED vertical structure (stability/CAPE, inversions, "
            "moisture, wind shear) at a future valid time. Unlike get_sounding, which is an "
            "OBSERVED sounding at 00/12Z, this projects the atmosphere forward. `station` is "
            "a 4-letter ICAO; `model` defaults to gfs (the only model with coverage outside "
            "North America); `fhr` is the forecast hour (0 = analysis; hourly early, then "
            "3-hourly). Coverage is dense over North America and sparse OCONUS -- an "
            "unavailable station is reported back so you can pick another."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KMSP"},
                "model": {"type": "string", "enum": list(fcstsounding.MODELS),
                          "description": "forecast model (default gfs)"},
                "fhr": {"type": "integer",
                        "description": "forecast hour (0=analysis; e.g. 6, 12, 24, 36)"},
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
            "Hourly MODEL point forecast TABLE for an airport: surface conditions over time "
            "-- temperature, dewpoint, wind, MSL pressure, low/mid/high cloud, and hourly "
            "precipitation at each forecast hour, from the model's BUFKIT output. Use it to "
            "see how conditions EVOLVE hour by hour at a site (complements get_fcst_sounding, "
            "which is the vertical profile at one hour). Each row is one forecast hour; read a "
            "column downward for a variable's trend. `station` 4-letter ICAO; `model` defaults "
            "to gfs (only gfs has OCONUS coverage); `hours` limits the horizon (default 48). "
            "Values are raw model surface fields."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KMSP"},
                "model": {"type": "string", "enum": list(fcstsounding.MODELS),
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
            "also visible, infrared, water_vapor. For a specific airport give its `station` "
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
                    "description": "satellite: geocolor (default)/visible/infrared/"
                    "water_vapor; radar: station_reflectivity/regional_mosaic/national_mosaic",
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
            "Fetch a short satellite LOOP (a time sequence of frames) centered on an airport, "
            "to see MOTION and TREND that a single still cannot show -- cloud advection, "
            "growth/erosion, fog burn-off, convective initiation. Returns a labeled filmstrip "
            "image (oldest to newest) and, for video-capable models, a short video. Give the "
            "`station` ICAO; optionally `product` (geocolor default/visible/infrared/"
            "water_vapor), `frames` (2-10, default 6), and `step_min` (minutes between frames, "
            "default 30). GOES (Americas) and Himawari (Japan/W Pacific) loops are wide-area; "
            "Meteosat (Europe/Africa/Middle East) loops are station-centered."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KWRI"},
                "product": {"type": "string", "enum": list(imagery.SAT_PRODUCTS),
                            "description": "geocolor (default)/visible/infrared/water_vapor"},
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
                          "description": "Optional single model; default shows all three."},
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
                    "stored step. Defaults to the earliest forecast hour.")},
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
            "How the archived model runs scored against OBSERVED METARs at the recent forecast "
            "hours leading up to your issue time -- per-hour forecast-vs-observed T/Td plus a "
            "mean bias per model. Exposes a run's warm/cold or dry/moist bias so you can weight "
            "the raw model output. Reads obs already in the store (leakage-safe), so it only "
            "covers hours at or before your issue time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "description": "4-letter ICAO, e.g. KMSP"},
                "model": {"type": "string", "enum": list(modeldata.MODELS),
                          "description": "Optional single model; default shows all three."},
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

TOOLS = [QUERY_OBS, GET_LATEST, GET_TREND, GET_SOUNDING, GET_MAP, GET_FCST_SOUNDING,
         GET_POINT_FORECAST, GET_CLIMO, GET_IMAGERY, GET_LOOP, GET_NEARBY_OBS, GET_TERRAIN,
         GET_MODEL_STATE, GET_HAZARD_SCAN, GET_MODEL_VERIFICATION, GET_NEARBY_MODEL_DATA,
         GET_CURRENT_TAF, CHECK_TAF]

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
            "cover values. Base the forecast only on the observations and trend provided."
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
            "TX/TN against the observed diurnal temperature range, and state the ONE "
            "hPa->inHg conversion you use everywhere), the TAF strategy, uncertainty, and "
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


def _get_sounding(args: dict) -> ToolResult:
    """Fetch an observed skew-T image from a public provider (network, no DB) and
    hand it back for the model to read. Site ids live in the provider's namespace,
    so a bad id/date surfaces as a fetch error the model can correct -- not a crash.
    The receipt cites the exact synoptic time + source URL (provenance)."""
    site = args.get("site")
    if not site:
        return ToolResult('error: get_sounding needs a "site" upper-air id, e.g. "site": "OUN"')
    source = str(args.get("source") or "spc").lower()
    if source not in ("spc", "wyoming"):
        return ToolResult(f'error: unknown source {source!r}; use "spc" or "wyoming"')
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


def _get_map(args: dict) -> ToolResult:
    """Fetch a catalogued surface/upper-air chart image (network, no DB). A forecast
    chart gets its GFS forecast hour snapped to the 6h grid; an unknown chart name or a
    fetch failure comes back as feedback, not a crash. Receipt cites the source URL."""
    name = args.get("chart")
    if not name or name not in wxmaps.CATALOG:
        return ToolResult(
            'error: get_map needs a valid "chart"; choose from: ' + ", ".join(wxmaps.CATALOG)
        )
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
        url = wxmaps.map_url(name, fhr=fhr, run=run)
        img = wxmaps.fetch_map(name, fhr=fhr, run=run)
    except Exception as e:  # noqa: BLE001 -- a fetch failure becomes feedback, not a dead loop
        # TT is third-party + URL-fragile; on failure fall back to the closest SPC
        # mesoanalysis ANALYSIS chart so the model keeps upper-air context. The receipt
        # states the degradation loudly -- it is NOW, not the forecast hour requested.
        fb = wxmaps.TT_TO_SPC_MESO.get(name) if spec.source == "tt" else None
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
    return ToolResult(
        f"{spec.label} [{name}]{lead} (source: {spec.source}, {url}); image follows.",
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
    # Meteosat takes an arbitrary bbox, so for a specific station we center a TIGHT local view
    # on it instead of the wide fixed region (the station-crop upgrade). Others use the sector.
    meteosat_point = center is not None and \
        imagery.SAT_REGIONS[region].provider == "meteosat_eumetsat_wms"
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


def _radar_degrade(icao: str, lat: float, lon: float, reason: str) -> ToolResult:
    """Fall back from a station-local view: the containing regional mosaic, then national.
    `reason` (guard miss or a station-fetch failure) is prepended so the receipt is honest.
    If the regional fetch itself fails (e.g. IEM down), continue to national -- which can
    degrade to the NWS GIF on a different host -- rather than dead-ending."""
    reg = imagery.radar_region_for_latlon(lat, lon)
    if reg:
        r = _radar_regional(reg)
        if r.images:                                  # regional succeeded
            r.text = (f"{reason}; showing the {imagery.RADAR_REGIONS[reg][1]} regional "
                      f"mosaic instead. {r.text}")
            return r
    nat = _radar_national("national radar mosaic")
    tail = ("regional mosaic also unavailable -- " if reg else
            f"{icao} is outside the curated radar regions -- ")
    nat.text = f"{reason}; {tail}showing the national mosaic for broad context only. {nat.text}"
    return nat


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

    # Honor an explicit mosaic choice directly -- do NOT route it through the guard (which
    # would fabricate a distance reason and hand back the wrong product).
    if product == "national_mosaic":
        return _radar_national("national radar mosaic (broad context only)")
    if product == "regional_mosaic":
        reg = imagery.radar_region_for_latlon(lat, lon)
        if reg:
            return _radar_regional(reg)
        nat = _radar_national("national radar mosaic")
        nat.text = (f"{icao} is outside the curated radar regions -- showing the national "
                    f"mosaic for broad context only. {nat.text}")
        return nat

    # Default / station_reflectivity: a station-centered local view when a radar is credible.
    near = imagery.nearest_radar(lat, lon)
    guard = imagery.RADAR_STATION_GUARD_KM
    if near and near[1] <= guard:
        site, dist = near
        try:
            url = imagery.radar_url("station", center=(lat, lon))
            img = imagery.fetch_radar("station", center=(lat, lon))
        except Exception as e:  # noqa: BLE001 -- degrade, don't dead-end (provider hiccup/outage)
            return _radar_degrade(icao, lat, lon,
                                  f"station radar fetch for {icao} failed ({type(e).__name__}: {e})")
        receipt = (f"Station-scale radar around {icao}, fetched {_fetch_stamp()} "
                   f"(nearest WSR-88D: {site['id']} {site['name']}, {dist:.0f} km; "
                   f"source: IEM NEXRAD composite, {url}); image follows.")
        return ToolResult(receipt, images=[img])
    reason = (f"nearest WSR-88D to {icao} is {near[1]:.0f} km away (beyond the {guard:.0f} km "
              f"local-radar guard)" if near else f"no radar site found near {icao}")
    return _radar_degrade(icao, lat, lon, reason)


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
    try:
        fr, source, coverage = imagery.satellite_loop(lat, lon, product,
                                                      frames=frames, step_min=step)
    except Exception as e:  # noqa: BLE001 -- no coverage / fetch failure -> feedback
        return ToolResult(f"error: could not build a satellite loop for {icao} "
                          f"({type(e).__name__}: {e}).")
    if len(fr) < 2:
        return ToolResult(f"error: only {len(fr)} loop frame(s) available for {icao}; "
                          "cannot show motion.")
    span = f"{fr[0][0]} -> {fr[-1][0]}"
    strip = charts.filmstrip(fr, title=f"{icao} {coverage} loop  {span}")
    mp4 = charts.loop_mp4(fr)
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


def _get_fcst_sounding(args: dict) -> ToolResult:
    """Fetch + render a model forecast sounding (network, no DB). A missing station or
    forecast hour comes back as feedback -- fcstsounding raises ValueError with the reason
    (404 / available hours) rather than crashing the loop. Receipt cites the source URL."""
    station = args.get("station")
    if not station:
        return ToolResult('error: get_fcst_sounding needs a "station" ICAO, e.g. "station": "KMSP"')
    model = str(args.get("model") or "gfs").lower()
    if model not in fcstsounding.MODELS:
        return ToolResult(f"error: unknown model {model!r}; choose from {', '.join(fcstsounding.MODELS)}")
    fhr = _int_arg(args.get("fhr"), 12, lo=0, hi=384)
    try:
        prof = fcstsounding.fetch_profile(station, model=model, fhr=fhr)
        png = charts.skewt(prof)
    except Exception as e:  # noqa: BLE001 -- fetch/parse failure becomes feedback, not a dead loop
        return ToolResult(f"error: could not build forecast sounding for {str(station).upper()} "
                          f"{model} f{fhr:03d} ({type(e).__name__}: {e})")
    receipt = (f"{model.upper()} forecast skew-T for {prof.station}, f{fhr:03d} valid "
               f"{prof.valid} (run {prof.run:%Y-%m-%dT%H:%MZ}, {prof.url}); image follows.")
    return ToolResult(receipt, images=[png])


def _uv_to_dirspd(u: float, v: float) -> tuple[int, int]:
    """Wind (u, v in m/s) -> (direction deg to nearest 10, speed kt). A presentation of the
    raw vector; the stored point-forecast data keeps the u/v components."""
    spd = round(math.hypot(u, v) * 1.94384)
    d = int(round((270.0 - math.degrees(math.atan2(v, u))) % 360.0 / 10.0) * 10) % 360
    return d, spd


def _fmt_point(pf, n: int) -> str:
    """Format a PointForecast as a text table: one row per forecast hour, columns are the
    raw surface variables (wind shown as dir/speed). Read a column down for a trend."""
    # Slice by valid TIME, not row count: the BUFKIT surface series is hourly early but
    # goes 3-hourly at longer ranges, so `n` rows would silently cover more than n hours.
    if pf.rows:
        cutoff = pf.rows[0]["valid"] + timedelta(hours=n)
        rows = [r for r in pf.rows if r["valid"] <= cutoff]
    else:
        rows = []
    out = [
        f"{pf.model.upper()} point forecast for {pf.station} -- run {pf.run:%Y-%m-%dT%H:%MZ}, "
        f"{len(rows)} hourly steps (source: {pf.url}). Raw model surface fields; each row is "
        "one forecast hour -- read a column down to see a variable's trend.",
        (f"{'Valid (UTC)':<18}{'T C':>5}{'Td C':>6}{'Wind kt':>10}{'MSLP':>7}"
         f"{'Cld L/M/H %':>14}{'P01 mm':>8}"),
    ]
    def _d(v, fmt: str = "{:.0f}") -> str:
        return "--" if v is None else fmt.format(v)

    for r in rows:
        u, v = r["uwnd_ms"], r["vwnd_ms"]
        wind = "--" if u is None or v is None else "{:03d}/{}".format(*_uv_to_dirspd(u, v))
        vt = f"{r['valid']:%Y-%m-%dT%H:%MZ}"
        trip = (r["lcld"], r["mcld"], r["hcld"])
        cloud = "--" if any(c is None for c in trip) else "/".join(f"{c:.0f}" for c in trip)
        out.append(
            f"{vt:<18}{_d(r['t2m_c']):>5}{_d(r['td2m_c']):>6}{wind:>10}"
            f"{_d(r['mslp_hpa']):>7}{cloud:>14}{_d(r['p01_mm'], '{:.1f}'):>8}"
        )
    return "\n".join(out)


def _get_point_forecast(args: dict) -> ToolResult:
    """Fetch + format a model point forecast table (network, no DB). A missing station (404)
    comes back as feedback via fcstsounding's ValueError, not a crash."""
    station = args.get("station")
    if not station:
        return ToolResult('error: get_point_forecast needs a "station" ICAO, e.g. "station": "KMSP"')
    model = str(args.get("model") or "gfs").lower()
    if model not in fcstsounding.MODELS:
        return ToolResult(f"error: unknown model {model!r}; choose from {', '.join(fcstsounding.MODELS)}")
    hours = _int_arg(args.get("hours"), 48, lo=1, hi=384)
    try:
        pf = fcstsounding.fetch_point(station, model=model)
    except Exception as e:  # noqa: BLE001 -- fetch/parse failure becomes feedback, not a dead loop
        return ToolResult(f"error: could not fetch point forecast for {str(station).upper()} "
                          f"{model} ({type(e).__name__}: {e})")
    return ToolResult(_fmt_point(pf, hours))


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
        return ToolResult(
            f"(no neighbor stations on file for {station}; the nearest-neighbor roster covers "
            "the benchmark's forecast stations)"
        )
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
    if unknown:
        out.append(f"(not in {station}'s fetchable roster, skipped: {', '.join(unknown)}; "
                   f"fetchable are: {', '.join(by_icao)})")
    for icao, dist, brg, de, _la, _lo in rows:
        head = f"{icao}  {dist:.0f} km {brg}  {de:+d} m"
        latest = store.latest(con, icao, 1)
        if not latest:
            out.append(f"{head}  | (no observation in store within the window)")
            continue
        r = latest[0]
        out.append(f"{head}  | {_decoded_line(r)}")
        out.append(f"    {r['raw']}")
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
            f"{'MSLP':>7}{'Cld%':>6}{'Vis':>6}{'Ceil ft':>9}",
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
    return "\n\n".join(blocks) + f"\n\nCROSS-MODEL: {synopsis}"


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
    ref = _pick_valid_time(piv["gfs"] or piv["hrrr"], want)
    if ref is None:
        return (f"(no pressure-level hazard data pre-fetched for {loc_id} -- prefetch runs with "
                f"hazards enabled for the site + grid only). {_md_locations_hint(con)}")
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


_VER_ALIASES = ("t2m", "td2m", "u10", "v10", "wind", "wdir")


def _fmt_model_verification(con, station: str, models: list[str]) -> str:
    lat, lon, _ = _resolve_md_location(con, station, None) or (None, None, None)
    if lat is None:
        return f"error: {station} is not a pre-fetched model-data location. {_md_locations_hint(con)}"
    # obs truth from the DB (leakage-safe: the per-run DB is cut off at issue time). Keyed by
    # nearest whole hour so a :53 METAR aligns to the top-of-hour model step.
    out = [f"Model-vs-obs verification for {station} -- archived forecast (from runs <= issue) "
           "vs observed METARs at the matching hours. Positive T/Td error = model too warm/moist.",
           ""]
    matched_any = False
    for model in models:
        rows = store.model_data_series(con, model, lat, lon, start=_WIDE_START, end=_WIDE_END,
                                       variables=list(_VER_ALIASES))
        series = _pivot_series(rows)
        if not series:
            continue
        lo, hi = series[0][0], series[-1][0]
        obs = {}
        for o in store.window(con, station, lo - timedelta(hours=1), hi + timedelta(hours=1)):
            key = (o["obs_time"] + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
            obs[key] = o
        run = next((r for _, r, _ in series if r), None)
        header = f"{model.upper()} (run {run:%Y-%m-%dT%HZ}):" if run else f"{model.upper()}:"
        block = [header, f"  {'Valid (Z)':<15}{'T f/o':>11}{'Terr':>6}{'Td f/o':>11}{'Tderr':>7}"]
        terrs = []
        for vt, _r, vm in series:
            o = obs.get(vt)
            if o is None:
                continue
            tf, tdf = _k2c(vm.get("t2m")), _k2c(vm.get("td2m"))
            to, tdo = o.get("temp_c"), o.get("dewpoint_c")
            te = f"{tf - to:+.1f}" if tf is not None and to is not None else "--"
            tde = f"{tdf - tdo:+.1f}" if tdf is not None and tdo is not None else "--"
            if tf is not None and to is not None:
                terrs.append(tf - to)
            tfo = f"{tf:.0f}/{to}" if tf is not None and to is not None else "--"
            tdfo = f"{tdf:.0f}/{tdo}" if tdf is not None and tdo is not None else "--"
            block.append(f"  {vt:%Y-%m-%dT%HZ}{tfo:>11}{te:>6}{tdfo:>11}{tde:>7}")
        if len(block) == 2:
            continue   # no overlapping obs for this model
        matched_any = True
        if terrs:
            block.append(f"  -> mean T bias {sum(terrs) / len(terrs):+.1f}C over {len(terrs)} hrs")
        out.append("\n".join(block))
        out.append("")
    if not matched_any:
        out.append("(no observed METARs overlap the archived forecast valid times yet -- "
                   "verification needs obs at the pre-issue forecast hours in the store)")
    return "\n".join(out)


def _get_model_verification(con, station: str, args: dict) -> ToolResult:
    model = args.get("model")
    models = [str(model).lower()] if model else list(modeldata.MODELS)
    return ToolResult(_fmt_model_verification(con, station, models))


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
             evidence_ids: list[str] | None = None) -> ToolResult:
    """Execute a model-issued tool call. The read tools run against a READ-ONLY
    connection; the sinks (emit_taf, check_taf, submit_taf_worksheet) and the network
    fetches (get_current_taf, get_sounding, get_map, get_fcst_sounding,
    get_point_forecast, get_imagery) need no DB and are handled first. `evidence_ids`
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
        return _stamp_fetched(_get_map(args))
    if name == "get_fcst_sounding":
        return _stamp_fetched(_get_fcst_sounding(args))
    if name == "get_point_forecast":
        return _stamp_fetched(_get_point_forecast(args))
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
        if name == "get_model_state":
            return _get_model_state(con, station, args)
        if name == "get_hazard_scan":
            return _get_hazard_scan(con, station, args)
        if name == "get_model_verification":
            return _get_model_verification(con, station, args)
        if name == "get_nearby_model_data":
            return _get_nearby_model_data(con, station, args)
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
