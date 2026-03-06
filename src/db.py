# src/db.py

import pandas as pd
from influxdb_client import InfluxDBClient

def connect(url, token, org):
    return InfluxDBClient(url=url, token=token, org=org)

def query_df(client, flux):
    result = client.query_api().query_data_frame(flux)
    if isinstance(result, list):
        if len(result) == 0:
            return pd.DataFrame()
        return pd.concat(result, ignore_index=True)
    return result



def get_series_by_subsystem(client, bucket, measurement, mission_id, subsystem):
    flux = f'''
from(bucket: "{bucket}")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "{measurement}")
  |> filter(fn: (r) => r.mission_id == "{mission_id}")
  |> filter(fn: (r) => r.subsystem == "{subsystem}")
  |> keep(columns: ["_time", "_value", "sensorId"])
'''
    df = query_df(client, flux)
    if df.empty:
        return df
    return df.rename(columns={"_time": "time", "_value": "value"}).sort_values("time")

