# Using xarray
# https://towardsdatascience.com/basic-data-structures-of-xarray-80bab8094efa

# Using Pandas
# https://pandas.pydata.org/docs/getting_started/10min.html

# Cartopy
# https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html

# Data
# https://search.earthdata.nasa.gov/search/granules?p=C1650311601-PODAAC&g=G1650319834-PODAAC&q=Oscar&m=-0.140625!0.0703125!2!1!0!0%2C2&tl=1567000344!4!!&fs10=Ocean%20Currents&fsm0=Ocean%20Circulation&fst0=Oceans

# Iterate Dict
# https://realpython.com/iterate-through-dictionary-python/

# Animation
# https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs

import cartopy.feature as cfeature
import shapely.geometry as sgeom
from datetime import datetime, timedelta

from cartopy.feature.nightshade import Nightshade
import netCDF4 as Dataset 
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import cartopy.geodesic as geo
import itertools
import pandas as pd
import matplotlib.animation as animation

import random
import math

_Num_Agents = 4
_Environments = {'currents':True,'solar':False}


class LowerThresholdRobinson(ccrs.Robinson):

    @property
    def threshold(self):
        return 1e3

class LowerThresholdPlateCarree(ccrs.PlateCarree):

    @property
    def threshold(self):
        return 0.01

class LowerThresholdGeodetic(ccrs.Geodetic):

    @property
    def threshold(self):
        return 0.01



def sample_America():
    """
    Returns lon and lat bounds for North America
    """
    lons = [-80, -66]

    lats = [28, 42]

    # [360-150, 360-50],[65, 20] zoomed out view. 360 for nasa data

    return lons, lats

def set_environment(dataset=None, lat_bnds = [42, 28], lon_bnds = [-80,-66]):
    '''
    This assumes that the georange is offest 20 and goes east from Greenwich.
    '''
    lon_bnds = list(map(lambda x: x + 360,lon_bnds))
    data = dataset.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds))
    data.attrs['NOTE2']='Nothing'
    data.attrs.pop('NOTE2',None)
    data.attrs['GEORANGE']=f'{lon_bnds[0]} to {lon_bnds[1]} {lat_bnds[0]} to {lat_bnds[1]}'
    return data


def sample_curr(data, shape=(3, 9)):
    """
    Return ``(x, y, u, v, crs)`` of some vector data
    computed mathematically. The returned crs will be in
    regular PlateCarree space.

    """
    crs = ccrs.PlateCarree()

    lons, lats = sample_America()
    lat_bnds, lon_bnds = [42, 28], [360-80,360-66]
    data = data.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds))

    x = np.linspace(lons[0], lons[1], len(data.longitude))
    y = np.linspace(lats[0], lats[1], len(data.latitude))

    x = data.longitude.values[:]
    y = data.latitude.values[:]

    u = data.u[0,0,:,:]
    v = data.v[0,0,:,:]
    print(u.shape)

    return x, y, u, v, crs

def plot_animation(environment, agents, lons, lats, Time_List):

    fig, ax = plt.subplots(figsize=(12, 6),
                       subplot_kw={'projection': LowerThresholdPlateCarree()})

    def animate(datetime_t, agents, environment):
        plt.cla()

        grid_lines =ax.gridlines(draw_labels=True)
        grid_lines.xformatter = LONGITUDE_FORMATTER
        grid_lines.yformatter = LATITUDE_FORMATTER

        ax.set_extent([-82, -60, 20, 44])
        ax.coastlines(resolution='10m') 

        ax.plot(lons, lats, transform=ccrs.PlateCarree(),color='m')
        ax.plot(lons, lats, transform=ccrs.Geodetic(),color='b')

        ax.set_title('US Costal Patrol\n' + f'({datetime64_2_datetime(datetime_t)})',pad=20)

        for agent_name in agents:
            agent_data = agents[agent_name].data.interp(time = datetime_t)
            agent_lon = agent_data.lon.values
            agent_lat = agent_data.lat.values
            agent_pose = ax.plot(agent_lon, agent_lat, 'o', transform=ccrs.PlateCarree(),color='r')
        
        # ax.plot(agent_lon, agent_lat, 'o', transform=ccrs.PlateCarree(),color='r')

        environment = environment.interp(time = datetime_t)
        # ax.add_feature(Nightshade(datetime64_2_datetime(datetime_t), alpha=0.2))
        magnitude = (environment.u.values[:,:] ** 2 + environment.v.values[:,:] ** 2) ** 0.5
        lon = environment.longitude.values
        lon[lon>180] = lon[lon>180] - 360
        ax.quiver(environment.longitude[:], environment.latitude[:], environment.u[:][:], environment.v[:][:], magnitude, transform=ccrs.PlateCarree())
        # ax.streamplot(lon, environment.latitude.values[:], environment.u.values[:,:], environment.v.values[:,:], density=2, color=magnitude, transform=ccrs.PlateCarree())
        ax.contourf(lon, environment.latitude.values[:], magnitude, transform=ccrs.PlateCarree(), alpha=0.2)
        # plt.colorbar()

    # ani = animation.FuncAnimation(fig, animate, frames=Time_List, interval=100, fargs=(agents,environment))
    ani = animation.FuncAnimation(fig, animate, frames=list(itertools.islice(Time_List, 0, len(Time_List), 8)), interval=10, fargs=(agents,environment))

    plt.show()

def plot_env(environment, agents, lons, lats, datetime_t):

    fig = plt.figure(figsize=(9,9))

    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax = fig.add_subplot(1, 1, 1, projection=LowerThresholdPlateCarree())

    grid_lines =ax.gridlines(draw_labels=True)
    grid_lines.xformatter = LONGITUDE_FORMATTER
    grid_lines.yformatter = LATITUDE_FORMATTER


    ax.set_extent([-82, -60, 20, 44], crs=ccrs.PlateCarree())


    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)

    ax.coastlines(resolution='10m') 

    # turn the lons and lats into a shapely LineString
    track = sgeom.LineString(zip(lons, lats))

    # buffer the linestring by two degrees (note: this is a non-physical
    # distance)
    track_buffer = track.buffer(2)

    ax.add_geometries([track_buffer], ccrs.PlateCarree(),
                      facecolor='#C8A2C8', alpha=0.5)
    ax.add_geometries([track], ccrs.PlateCarree(),
                      facecolor='none', edgecolor='k', alpha=0.5)

    # Add agent positions
    for agent_name in agents:
        agent_data = agents[agent_name].data.interp(time = datetime_t)
        agent_lon = agent_data.lon.values
        agent_lat = agent_data.lat.values
        ax.plot(agent_lon, agent_lat, 'o', transform=ccrs.PlateCarree(),color='r')


    # ax.plot(lons[0], lats[0], 'o', transform=ccrs.PlateCarree(),color='g')
    ax.plot(lons, lats, transform=ccrs.PlateCarree(),color='m')
    # ax.plot(lons, lats, transform=LowerThresholdGeodetic(),color='b', markevery=1)
    ax.plot(lons, lats, transform=ccrs.Geodetic(),color='b')

    # Correct environment dimensions by selecting a single date. Interpolate between times of consecuative datasets.
    # environment = environment.sel(time = datetime_t)
    environment = environment.interp(time = datetime_t)


    ax.set_title('US Costal Patrol\n' + f'({datetime64_2_datetime(datetime_t)})',pad=20)
    # ax.set_title('US Costal Patrol\n' + f'({environment.time.values})',pad=20)
    ax.add_feature(Nightshade(datetime64_2_datetime(datetime_t), alpha=0.2))

    # ax.quiver(environment.longitude[:], environment.latitude[:], environment.u[0][0][:][:], environment.v[0][0][:][:], transform=ccrs.PlateCarree())
    # ax.quiver(environment.longitude[:], environment.latitude[:], environment.u[:][:], environment.v[:][:], transform=ccrs.PlateCarree())
    magnitude = (environment.u.values[:,:] ** 2 + environment.v.values[:,:] ** 2) ** 0.5
    # print(magnitude)
    # ax.streamplot(environment.longitude.values[:], environment.latitude.values[:], environment.u.values[:,:], environment.v.values[:,:], density=2, color=magnitude, transform=ccrs.PlateCarree())

    plt.show()
    return 1

def m_s_2_knots(m_s):
    return m_s * 1.94384

def knots_2_m_s_(knots):
    return knots * 0.514444

def get_start_date(dataset):
    '''
    Return numpy datetime object of first data entry
    '''
    start_date = dataset.time[0].values
    return start_date

def datetime64_2_datetime(npdatetime):
    '''
    Return datetime object from npdatetime object
    '''
    ns = 1e-9 # number of seconds in a nanosecond
    date = datetime.utcfromtimestamp(npdatetime.astype(int) * ns)
    return date

def fetch_datasets(lons, lats):
    dataset = xr.open_dataset("data/processed/oscar_vel_18_20.nc").sel(longitude=slice(*lons), latitude=slice(*lats))
    dataset = dataset.drop(['year'])
    dataset = dataset.sel(depth = 0, method = 'nearest')
    return dataset

def get_geo_inverse(earth, point1, point2):
    result_inverse = earth.inverse(point1,point2)
    result_inverse = np.asarray(result_inverse)
    distance = result_inverse[0][0]
    forward_azimuth_start = result_inverse[0][1]
    return distance, forward_azimuth_start

def get_geo_direct(earth, point1, a1, step_distance):
    result_direct = earth.direct(point1,a1, step_distance)
    result_direct = np.asarray(result_direct)
    long_step_goal = result_direct[0][0]
    lat_step_goal = result_direct[0][1]
    a1 = result_direct[0][2]
    return long_step_goal, lat_step_goal, a1


def get_route(earth):
    lons, lats = sample_America()
    start_point = [lons[0],lats[0]]
    end_point = [lons[1],lats[1]]
    distance, forward_azimuth_start = get_geo_inverse(earth, start_point, end_point)
    #      [Lat,Long]  [Lat,Long] (m)     (degrees)
    return start_point, end_point, distance, forward_azimuth_start

def get_goal_list(ave_speed_goal, earth, distance, start_point, endpoint, forward_azimuth_start):
    '''
    Calculate a list of goal points evenly spaced between the start and end track points for a certain goal speed of transit.
    Assumes hourly updates.
    '''
    goal_list = []
    point1 = start_point
    goal_list.append(start_point)
    point2 = endpoint
    a1 = forward_azimuth_start #course ship takes at start to drive towards end point along great circle 
    ave_speed_goal = knots_2_m_s_(ave_speed_goal) # m/s
    ave_speed_goal = ave_speed_goal*3600 # m/hr
    # use floor to go slightly faster on average
    subdivisions = math.floor(distance/ave_speed_goal) # num of hours to reach goal = num of timesteps
    ave_speed = distance/subdivisions
    print(f'Average Speed: {ave_speed/3600} m/s, {m_s_2_knots(ave_speed/3600)} knots')
    step_distance = ave_speed 
    print(f'Step Distance: {step_distance}')
    for step in range(subdivisions):
        long_goal, lat_goal, _ = get_geo_direct(earth, point1, a1, step_distance)
        goal_list.append([long_goal, lat_goal])
        point1 = [long_goal, lat_goal]
        _, a1 = get_geo_inverse(earth, point1, point2)
    return goal_list, subdivisions


class Agent:
    def __init__(self, init_time = None, pose = [None,None,None], goal = None):
        self.batt = 100
        self.H2 = 100
        self.released = False
        self.init_time = init_time
        self.pose = pose # long, lat, heading
        self.goal = goal
        self.goal_list = None
        self.goal_pos_counter = None
        self.start_offset = None
        self.time = None
        self.history = {'time':[], 'pose_lon':[], 'pose_lat':[], 'goal_lon':[], 'goal_lat':[], 'batt':[], 'H2':[]}
        self.data = None
    def setgoal(self, time, route = 'straight', goal_list = None, start_offset = 0):
        self.time = time
        if route == 'straight':
            if goal_list == None: # ie not passed a list
                lons, lats = sample_America()
                self.pose = [lons[0],lats[0]]
                self.goal = [lons[1],lats[1]]
            else:
                self.goal_list = goal_list
                self.start_offset = start_offset
                lon_p, lat_p = self.goal_list[self.start_offset][:]
                self.pose = [lon_p,lat_p]
                # If the start offset is at the end of the goal list (turn around point) then the next goal should point back towards start.
                # Otherwise, the goal should be the next item in goal list and goal_pos_count should be this index
                if self.start_offset+1 < len(self.goal_list):
                    self.goal_pos_counter = itertools.count(self.start_offset+1)
                else:
                    self.goal_pos_counter = itertools.count(self.start_offset-1,-1)
                goal_count = next(self.goal_pos_counter)
                lon_g, lat_g = self.goal_list[goal_count][:]
                self.goal = [lon_g,lat_g]
        else:
            print('IMPLIMENT INTELLIGENCE')
        self.append_history()
        return 1
    def update(self, time):
        self.time = time
        self.pose = self.goal
        goal_count = next(self.goal_pos_counter)
        # If goal count < 0, then we are back home and need to turn back to far goal
        if goal_count < 0:
            print('MADE HOME')
            self.goal_pos_counter = itertools.count(1)
            goal_count = next(self.goal_pos_counter)
        # OTW if goal count is < [len(goal_list) - 1], keep going
        elif goal_count+1 < len(self.goal_list):
            t = 3
        # OTW time to turn back toward home     
        else:
            print('MADE FAR')
            self.goal_pos_counter = itertools.count(len(self.goal_list)-1,-1)
            goal_count = next(self.goal_pos_counter)
        # print(f'Len Goal List: {len(self.goal_list)}')
        # print(f'Goal Count Index: {goal_count}')
        lon_g, lat_g = self.goal_list[goal_count][:]
        self.goal = [lon_g,lat_g]
        self.append_history()
        return 1
    def append_history(self):
        self.history['time'].append(self.time)
        self.history['pose_lon'].append(self.pose[0])
        self.history['pose_lat'].append(self.pose[1])
        self.history['goal_lon'].append(self.goal[0])
        self.history['goal_lat'].append(self.goal[1])
        self.history['batt'].append(self.batt)
        self.history['H2'].append(self.H2)
        return 1
    def store_data(self):
        times = self.history['time']
        pose_lon = self.history['pose_lon']
        pose_lat = self.history['pose_lat']
        ds = xr.Dataset(data_vars={"lon":(["time"],pose_lon), 
                               "lat":(["time"],pose_lat)}, 
                    coords={"time": times})
        self.data = ds
        return 1
def set_agents_deployment(agents, start_datetime, transit_leg_hrs):
    hrs_spacing = transit_leg_hrs/len(agents)
    for i_key, agent in enumerate(agents):
        print(f'Enumerate Agents\n{i_key}, {agent}')
        # print(i_key * hrs_spacing)
        # print(np.timedelta64(int(i_key * hrs_spacing),'h'))
        agents[agent].init_time =start_datetime + np.timedelta64(int(i_key * hrs_spacing),'h')
    # agents['Agent_0'].init_time = start_datetime
    # print(agents['Agent_0'].init_time)
    # agents['Agent_1'].init_time = start_datetime + np.timedelta64(10,'D') # Time range of simulation
    

def main():
    agents = {}
    print(f'\nNumber of Agents: {_Num_Agents}')
    print(f'Environment Model: {_Environments}')
    

    # Fetch the parent dataset
    environment_dataset = fetch_datasets([360-150, 360-50],[65, 20])


    # Initialize Start Time, Stop Time, and Time Interval
    start_datetime = get_start_date(environment_dataset)
    datetime_t = start_datetime # current time counter that increases with simulation
    print()
    print(f'Start Date: {start_datetime}')

    date_range = np.timedelta64(100,'D') # Time range of simulation
    stop_datetime = start_datetime + date_range
    print(f'Stop Date: {stop_datetime}')
    print()


    # Build earth
    earth = geo.Geodesic()

    # Get route info
    start_point, endpoint, transit_distance, forward_azimuth_start = get_route(earth)
    ave_speed = 2 # knots
    goal_list, subdivisions = get_goal_list(ave_speed, earth, transit_distance, start_point, endpoint, forward_azimuth_start)
    transit_leg_hrs = subdivisions
    print(f'Transit Distance: {transit_distance/1000} km, Start [Lon,Lat]: {start_point}, End [Lon,Lat]: {endpoint}, Initial Heading: {forward_azimuth_start}')
    print() 

    # Initialize local environment and agents
    # environment = set_environment(start_datetime, environment_dataset)
    environment = set_environment(environment_dataset)
    print('Environment___________')
    print(environment)

    # Build Agents
    print('Building Agents')
    for agentname in range(_Num_Agents):
        agents[f'Agent_{agentname}'] = Agent()
    print(list(agents.keys()))


    # conda install -c conda-forge requests 

    # date = datetime(1999, 12, 31, 12)
    

    # print(environment)
    # print(environment[{'time': environment['time'] == np.datetime64(datetime_t)}]['time'])
    # print(environment.drop('year')['time'])

    # MUST UPDATE TO INCLUDE LOOPING THOUGH AGENTS AND SETTING START TIMES. OTW OTHER AGENTS WOULD BREAK CODE!
    agents['Agent_0'].init_time = start_datetime
    print(agents['Agent_0'].init_time)
    agents['Agent_1'].init_time = start_datetime + np.timedelta64(10,'D') # Time range of simulation
    set_agents_deployment(agents, start_datetime, transit_leg_hrs)

    simulation_timestep = np.timedelta64(1,'h')
    # Time_List = np.arange(start_datetime,stop_datetime,simulation_timestep)
    Time_List = np.arange(start_datetime,start_datetime + np.timedelta64(transit_leg_hrs+1,'h'),simulation_timestep)

    for datetime_t in Time_List:
        for agent in agents:
            if agents[agent].released:
                # print(f'Datetime: {datetime_t}')
                # print(f'Update {agent}')
                agents[agent].update(datetime_t)
                # print(f'Pose: {agents[agent].pose} \nGoal: {agents[agent].goal}')
            if agents[agent].init_time <= datetime_t and agents[agent].released == False:
                agents[agent].released = True
                agents[agent].setgoal(datetime_t, route = 'straight', goal_list = goal_list, start_offset = 0)
                print(f'Release {agent}')

    # Store data
    for agent in agents:
        agents[agent].store_data()
        print(f'\n{agent} Data Xarray: \n{agents[agent].data}')

    for i, datetime_t in enumerate(Time_List):
        if i%240 == 0:
            # plot_env(environment, agents, *sample_America(), datetime_t)
            pass
            
    # plot_env(environment, agents, *sample_America(), datetime_t)

    plot_animation(environment, agents, *sample_America(), Time_List)


   


    # times = agents['Agent_0'].history.pop('time')
    # df = pd.DataFrame(agents['Agent_0'].history, index = times)
    # df = pd.DataFrame([agents['Agent_0'].history['pose_lon'],agents['Agent_0'].history['pose_lat']])
    # print(df)
    # ds = xr.Dataset(df)
    # print(ds)
    # print(ds['pose_lon'])
    # ds['pose_lon'].plot.line()
    # plt.show()
if __name__ == '__main__':
    main()
