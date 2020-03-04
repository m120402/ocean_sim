import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Any import of metpy will activate the accessors
import metpy.calc as mpcalc
# from metpy.testing import get_test_data
from metpy.units import units

# https://stackoverflow.com/questions/58474640/how-to-plot-ocean-currents-with-cartopy

# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.streamplot.html#matplotlib.axes.Axes.streamplot

# with xr.open_dataset('data/oscar_vel9994.nc') as ds:
#     print(ds)

#     ax = plt.axes(projection=ccrs.PlateCarree())

#     dec = 10
#     lon = ds.longitude.values[::dec]
#     lon[lon>180] = lon[lon>180] - 360
#     mymap=plt.streamplot(lon, ds.latitude.values[::dec], ds.u.values[0, 0, ::dec, ::dec], ds.v.values[0, 0, ::dec, ::dec], 6, transform=ccrs.PlateCarree())
#     ax.coastlines()
#     plt.show()

with xr.open_dataset('data/oscar_vel9994.nc') as ds:
    # print()
    print(ds.u)
    # print(ds.longitude)
    lat_bnds, lon_bnds = [42, 28], [360-80,360-66]
    data = ds.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds))
    print(data.u)
    # print(ds[dict()])

    # To parse the full dataset, we can call parse_cf without an argument, and assign the returned
    # Dataset.
    # data = ds.metpy.parse_cf()

    # ax = plt.axes(projection=ccrs.PlateCarree())

    # lons = [-80, -66]

    # lats = [28, 42]

    # dec = 10
    # lon = ds.longitude.values[::dec]
    # print(lon)
    # print(lon.shape)
    # print()
    # lon[lon>180] = lon[lon>180] - 360
    # print(lon)
    # print(lon.shape)
    # print()


    # mymap=plt.streamplot(lon, ds.latitude.values[::dec], ds.u.values[0, 0, ::dec, ::dec], ds.v.values[0, 0, ::dec, ::dec], 6, transform=ccrs.PlateCarree())
    # # mymap=plt.streamplot(lon, lat, ds.u.values[0, 0, ::dec, ::dec], ds.v.values[0, 0, ::dec, ::dec], 6, transform=ccrs.PlateCarree())
    
    magnitude = (data.u.values[0,0,:,:] ** 2 + data.v.values[0,0,:,:] ** 2) ** 0.5

    lon = data.longitude.values
    lon[lon>180] = lon[lon>180] - 360
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-82, -60, 20, 44], crs=ccrs.PlateCarree())
    mymap=plt.streamplot(lon, data.latitude.values[:], data.u.values[0,0,:,:], data.v.values[0,0,:,:], 4, color=magnitude, transform=ccrs.PlateCarree())
    # magnitude.plot()
    plt.contourf(lon, data.latitude.values[:], magnitude, transform=ccrs.PlateCarree(), alpha=0.2)
    ax.coastlines()

    plt.show()

    print(magnitude.shape)


    # lon_p = lon[lons[0]:lons[1]]
    # print(lon_p)
    # lat_p = ds.latitude.values[lats[0]:lats[1]]   
    # print(lat_p) 
    # print('lon: ',lon_p.shape)
    # print('lat: ',lat_p.shape)
    # print('lon: ',lon.shape)
    print('latitude: ',data.latitude.values[:].shape)
    print('longitude: ',data.longitude.values[:].shape)
    print('u: ',data.u.values.shape)
    print('v: ',data.v.values.shape)
    print('u: ',data.u.values[0, 0, :, :].shape)
    print('v: ',data.v.values[0, 0, :, :].shape)

    print('latitude: ',data.latitude.values[:])
    print('longitude: ',data.longitude.values[:])
    print('u: ',data.u.values[0, 0, :, :])
    print('v: ',data.v.values[0, 0, :, :])


    # lon_p = lon[lons[0]:lons[1]]
    # print(lon_p)
    # lat_p = ds.latitude.values[lats[0]:lats[1]]   
    # print(lat_p) 
    # print('lon: ',lon_p.shape)
    # print('lat: ',lat_p.shape)
    # print('lon: ',lon.shape)
    # print('latitude: ',ds.latitude.values.shape)
    # print('u: ',ds.u.values.shape)
    # print('v: ',ds.v.values.shape)
    # print('u: ',ds.u.values[0, 0, ::dec, ::dec].shape)
    # print('v: ',ds.v.values[0, 0, ::dec, ::dec].shape)


    # ax.set_extent([-82, -60, 20, 44], crs=ccrs.PlateCarree())

    # dec = 10
    # lon = ds.longitude.values[::dec]
    # print(len(lon))
    # print(ds.longitude)
    # print()
    # print(ds.u.values.shape)

