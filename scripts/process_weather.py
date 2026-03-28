import xarray as xr
import numpy as np
import pandas as pd

#load files
ds1 = xr.open_dataset("../data/file1.nc")
ds2 = xr.open_dataset("../data/file2.nc")

#Merge datasets
ds = xr.concat([ds1,ds2], dim="time")


#Extraxt variables
temp = ds['t2m']
dew = ds['d2m']
u_wind = ds['u10']
v_wind = ds['v10']
rain = ds['tp']

#Derived features
wind_speed = np.sqrt(u_wind**2 + v_wind**2)
humidity = temp - dew

#Reduce time dimension
temp_avg = temp.mean(dim=["time", "valid_time"], skipna=True)
humidity_avg = humidity.mean(dim=["time", "valid_time"], skipna=True)
wind_avg = wind_speed.mean(dim=["time", "valid_time"], skipna=True)
rain_avg = rain.mean(dim=["time", "valid_time"], skipna=True)
rain_mm = rain_avg*1000

# Convert to arrays
temp_arr = temp_avg.values
humidity_arr = humidity_avg.values
wind_arr = wind_avg.values
rain_arr = rain_avg.values

#print(temp_avg.values)
#print(humidity_avg.values)
#print(wind_avg.values)
#print(rain_mm.values)

lat = ds.latitude.values
lon = ds.longitude.values

data = []
for i in range(len(lat)):
    for j in range(len(lon)):
        data.append([
            lat[i],
            lon[j],
            temp_arr[i][j],
            humidity_arr[i][j],
            wind_arr[i][j],
            rain_arr[i][j]
            ])
df = pd.DataFrame(data, columns=[
    "latitude", "longitude",
    "temperature", "humidity",
    "wind_speed", "rainfall"
    ])
print(df.head())

df.to_csv("../data/weather_features.csv", index = False)

