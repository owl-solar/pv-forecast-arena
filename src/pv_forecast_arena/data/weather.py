"""Open-Meteo weather.

- Historical Forecast / Previous Runs API -> archived forecasts (used for training and test).
- Archive API -> observed/reanalysis weather (only for the "oracle" comparison run).

All timestamps in UTC. Cache every download to data/raw as parquet.
"""
