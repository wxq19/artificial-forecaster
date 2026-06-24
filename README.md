# Artificial Forecaster
## Goal:
Test whether a multi-modal language model (some kind of vision language model) can replicate Air Force Weather forecaster tasks. It needs to do the following:

- Ingest forecasting data
  - METARs
  - TAFs
  - NWP GRIB files
  - Satellite imagery
  - Radar imagery
  - Forecasting charts
  - Skew-Ts
  - Climatology
  - SIGMETs/PIREPs
  - Miscellaneous forecasting products
- Reason through weather forecasts
- Create forecasting products
  - TAFs
  - MEFs
  - WWAs

## Design 
TBD

## Tech Stack
Dependencies managed with `uv`. Run scripts with `uv run python ...`. 

### Geospatial Tools
TBD

## Project Structure
```
artificial-forecaster/
├── .env                  # real config + keys — GITIGNORED, never commit
├── .env.example          # template with blank values (committed)
├── pyproject.toml        # has [tool.hatch.build.targets.wheel] + [tool.uv] package=true
├── src/forecaster/
│   ├── config.py         # typed settings, reads .env (the ONLY config source)
│   └── llm.py            # the ONLY file that builds the OpenAI client
└── test_endpoint.py      # text + vision test
```