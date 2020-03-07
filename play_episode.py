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

import sys
sys.path.append('../programs')

import monoHFC as design
import pickle


_Num_Agents = 1
_Environments = {'currents':True,'solar':False}
_Write = False


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

def knots_2_m_s(knots):
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
    ave_speed_goal = knots_2_m_s(ave_speed_goal) # m/s
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

class plant():
    def __init__(self):
        # Maxeon 400 W
        # https://static1.squarespace.com/static/5354537ce4b0e65f5c20d562/t/5cf6764c6002390001d39a4d/1559656028949/SunPower+Maxeon+3-400-390-370-au_0+Specification+Sheet+2019.pdf
        # https://www.solar-electric.com/learning-center/solar-charge-controller-basics.html/
        self.solar_rating = 400/(1.69*1.046) # W/m2 @ STC
        self.MPPT_eff = 0.95 # maximum power point tracking
        self.inverter_eff = 0.98 # Inverter efficiency
        self.STC_irradiance = 1000 # Standard Test Conditions (1000 W/m2 irradiance, AM 1.5, 25° C)
        self.hotel_load = 4.1035 # kW
        self.LHV_H2 = 120.21 #MJ/kg
        self.HFC_efficiency = 0.69 # Efficiency at 20% load
        self.H2_Max = 427.5 # Kg of hydrogen gas at 500 bar (stored in 300L canisters)
        self.Batt_Max = 675.0 # kWh
        self.hrs_effective_solar = 5 # 5 hours of rated power daily
        self.boat = None
        self.max_solar_power_in = None
        self.max_solar_daily_average = None

    def get_boat(self):
        self.boat = design.Opt_Hull()
        x0 = [50]
        sol = self.boat.minima(x0)
        # self.boat = sol
        self.hotel_load = self.boat.hfc.HotelLoads
        self.solar_area = self.boat.deckArea
        self.max_solar_power_in = self.boat.solar.solar_area * self.solar_rating/1000 # kW
        self.max_solar_daily_average = self.max_solar_power_in * self.hrs_effective_solar * self.MPPT_eff * self.inverter_eff #kWh
        print(f'MAX SOLAR DAILY AVERAGE: {self.max_solar_daily_average} kWh')
class Agent():
    def __init__(self, earth = geo.Geodesic(), init_time = None, pose = [None,None,None], goal = None):
        # Init Stuff
        self.earth = earth
        self.released = False
        self.init_time = init_time
        self.goal_list = None
        self.goal_pos_counter = None
        self.start_offset = None
        self.plant = plant()
        self.plant.get_boat()

        # State
        self.pose = pose # long, lat, heading
        self.goal = goal
        self.time = None 
        self.batt = None
        self.H2 = None
        self.heading = None
        self.STW = None
        self.u = None
        self.v = None
        self.irradiance = None

        # Data Storage
        self.history = {'time':[], 'pose_lon':[], 'pose_lat':[], 'goal_lon':[], 'goal_lat':[], 'solar':[], 'batt':[], 'H2':[], 'heading':[], 'STW':[], 'irradiance':[], 'u':[], 'v':[]}
        self.data = None
    def action(self, distance, course):
        minval = 0.00001
        u = self.u
        v = self.v
        u_goal = distance * math.cos(math.radians(course))/3600
        v_goal = distance * math.sin(math.radians(course))/3600

        u_req = u_goal - u
        v_req = v_goal - v
        den = v_req if abs(v_req) > minval else minval
        heading = math.degrees(math.atan2(u_req,den))
        STW = np.hypot(u_req,v_req)
        return heading, STW
    def propulsion_power(self):
        speed_kts = m_s_2_knots(self.STW)
        _, power = self.plant.boat.set_res_and_power_holtrop(speed_kts)
        # print(f'Speed: {speed_kts}\nPower: {power/1000}')
        return power/1000
    def get_irradiance(self,hour, irradiance):
        pass

    def setgoal(self, time, route = 'straight', goal_list = None, start_offset = 0, currents = [0,0], irradiance = 1000):
        self.time = time
        pd_time = pd.to_datetime(time)
        hour = pd_time.hour
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
        # Start Full
        self.batt, self.H2 = 100, 100
        distance, course = get_geo_inverse(self.earth, self.pose, self.goal)
        self.u, self.v = currents.u.values, currents.v.values
        self.irradiance = irradiance
        self.heading, self.STW = self.action(distance, course)
        # print(f'FIRST COURSE = {course}')
        self.append_history()
        return 1
    def update(self, time, currents = [0,0], irradiance = 1000):
        self.time = time
        pd_time = pd.to_datetime(time)
        hour = pd_time.hour
        self.pose = self.goal
        goal_count = next(self.goal_pos_counter)
        # If goal count < 0, then we are back home and need to turn back to far goal
        if goal_count < 0:
            # print('MADE HOME')
            self.goal_pos_counter = itertools.count(1)
            goal_count = next(self.goal_pos_counter)
        # OTW if goal count is < [len(goal_list) - 1], keep going
        elif goal_count+1 < len(self.goal_list):
            t = 3
        # OTW time to turn back toward home     
        else:
            # print('MADE FAR')
            self.goal_pos_counter = itertools.count(len(self.goal_list)-1,-1)
            goal_count = next(self.goal_pos_counter)
        lon_g, lat_g = self.goal_list[goal_count][:]
        self.goal = [lon_g,lat_g]

        # Use old environmental contitions to calculate plant state
        self.transition() #sets solar, batt, H2
        # Set New environmental conditions for next time step
        distance, course = get_geo_inverse(self.earth, self.pose, self.goal)
        self.u, self.v = currents.u.values, currents.v.values
        self.irradiance = irradiance
        self.heading, self.STW = self.action(distance, course)
        # print(f'FIRST COURSE = {course}')
        self.append_history()
        return 1
    def transition(self):
        '''
        Calculates the solar % rating, Battery % charge, and H2 % fuel using previous timesteps
        state [irradiance, batt, H2, STW]
        '''
        # Assumed until later updated to be dynamic
        time_unit = np.timedelta64(1,'h')
        hours_passed = (time_unit/np.timedelta64(1,'h'))

        propulsion_power = self.propulsion_power()
        load_power = propulsion_power + self.plant.hotel_load # kW
        load_energy_out = load_power * hours_passed # kWh
        # Later account for Temp STC adjustment
        solar_in = self.plant.max_solar_power_in * (5/24) * (self.irradiance / self.plant.STC_irradiance)
        batt_in = solar_in * self.plant.MPPT_eff
        # Later modulate max solar daily average by actual STC irradiance fraction
        max_batt_out = (self.plant.max_solar_daily_average / 24)* hours_passed
        if (load_energy_out / self.plant.inverter_eff) > max_batt_out:
            batt_out = max_batt_out 
        else:
            batt_out = load_energy_out / self.plant.inverter_eff
        if self.batt <= 50:
            batt_out = 0
        HFC_out = max(0,(load_energy_out - batt_out * self.plant.inverter_eff) / self.plant.inverter_eff) #kWh
        H2_out = HFC_out * 3600 / (1000 * self.plant.LHV_H2) / self.plant.HFC_efficiency
        self.H2 -= (H2_out/self.plant.H2_Max)
        self.H2 = max(0, self.H2)
        batt_net = batt_in - batt_out
        batt_per_change = batt_net / self.plant.Batt_Max
        self.batt += batt_per_change
        self.batt = min(100, self.batt)
        # print()
        # print(f'max_solar_power_in: {self.plant.max_solar_power_in}')
        # print(f"solar_in: {solar_in}")
        # print(f"batt_in: {batt_in}")
        # print(f"load_energy_out: {load_energy_out}")
        # print(f"max_batt_out: {max_batt_out}")
        # print(f'Self_Batt: {self.batt}')
        # print(f'Batt Out: {batt_out}')
        # print(f'HFC_out: {HFC_out}')
        # print(f'H2_out: {H2_out}')
        # print(f'Self H2: {self.H2}')

    def append_history(self):
        self.history['time'].append(self.time)
        self.history['pose_lon'].append(self.pose[0])
        self.history['pose_lat'].append(self.pose[1])
        self.history['goal_lon'].append(self.goal[0])
        self.history['goal_lat'].append(self.goal[1])
        self.history['batt'].append(self.batt)
        self.history['H2'].append(self.H2)
        self.history['heading'].append(self.heading)
        self.history['STW'].append(self.STW)
        self.history['u'].append(self.u)
        self.history['v'].append(self.v)
        self.history['irradiance'].append(self.irradiance)
        return 1
    def store_data(self):
        times = self.history['time']
        pose_lon = self.history['pose_lon']
        pose_lat = self.history['pose_lat']
        goal_lon = self.history['goal_lon']
        goal_lat = self.history['goal_lat']
        batt = self.history['batt']
        H2 = self.history['H2']
        heading = self.history['heading']
        STW = self.history['STW']
        u = self.history['u']
        v = self.history['v']
        irradiance = self.history['irradiance']
        ds = xr.Dataset(data_vars={"lon":(["time"],pose_lon), 
                               "lat":(["time"],pose_lat), 
                               "goal_lon":(["time"],goal_lon), 
                               "goal_lat":(["time"],goal_lat), 
                               "batt":(["time"],batt), 
                               "H2":(["time"],H2), 
                               "heading":(["time"],heading), 
                               "STW":(["time"],STW), 
                               "u":(["time"],u), 
                               "v":(["time"],v), 
                               "irradiance":(["time"],irradiance)}, 
                    coords={"time": times})
        self.data = ds
        return 1
def set_agents_deployment(agents, start_datetime, transit_leg_hrs):
    hrs_spacing = transit_leg_hrs*2/len(agents)
    for i_key, agent in enumerate(agents):
        agents[agent].init_time =start_datetime + np.timedelta64(int(i_key * hrs_spacing),'h')    

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
    # print('Environment___________')
    # print(environment)

    # Build Agents
    print('Building Agents:')
    for agentname in range(_Num_Agents):
        agents[f'Agent_{agentname}'] = Agent()
    print(list(agents.keys()))


    # conda install -c conda-forge requests 

    # date = datetime(1999, 12, 31, 12)
    
    # MUST UPDATE TO INCLUDE LOOPING THOUGH AGENTS AND SETTING START TIMES. OTW OTHER AGENTS WOULD BREAK CODE!
    set_agents_deployment(agents, start_datetime, transit_leg_hrs)

    simulation_timestep = np.timedelta64(1,'h')
    # Time_List = np.arange(start_datetime,stop_datetime,simulation_timestep)
    # Time_List = np.arange(start_datetime,start_datetime + np.timedelta64(transit_leg_hrs+1,'h'),simulation_timestep)
    Time_List = np.arange(start_datetime,start_datetime + np.timedelta64(365*2,'D'),simulation_timestep)

    # _Write = True
    if _Write:
        for datetime_t in Time_List:
            for agent in agents:
                if agents[agent].released:
                    # currents = environment.interp(time = datetime_t, longitude = agents[agent].pose[0] + 360, latitude = agents[agent].pose[1])                
                    currents = environment.sel(time = datetime_t, longitude = agents[agent].pose[0] + 360, latitude = agents[agent].pose[1], method='nearest')                
                    # agents[agent].update(datetime_t)
                    agents[agent].update(datetime_t, currents= currents)
                if agents[agent].init_time <= datetime_t and agents[agent].released == False:
                    agents[agent].released = True
                    currents = environment.interp(time = datetime_t, longitude = start_point[0] + 360, latitude = start_point[1])
                    agents[agent].setgoal(datetime_t, route = 'straight', goal_list = goal_list, start_offset = 0, currents= currents)
                    print(f'Release {agent}')

        # Store data
        for agent in agents:
            agents[agent].store_data()
            # print(f'\n{agent} Data Xarray: \n{agents[agent].data}')

        for i, datetime_t in enumerate(Time_List):
            if i%240 == 0:
                # plot_env(environment, agents, *sample_America(), datetime_t)
                pass
    
    print()
    print('Arrived Here')

    # Pickle agents and their history:
    if _Write:
        with open('agents.pickle', 'wb') as w:
            pickle_agents = {}
            print('Dumping Pickle')
            for agent in agents:
                pickle_agents[agent] = Agent(earth = None)
                pickle_agents[agent].data = agents[agent].data
            pickle.dump(pickle_agents, w)

    with open('agents.pickle', 'rb') as r:
        # pickle.dump(agents['Agent_0'].data, f)
        agents = pickle.load(r)
        print('\nWant a pickle?')
        print(agents)

    # plot_env(environment, agents, *sample_America(), datetime_t)

    plot_animation(environment, agents, *sample_America(), Time_List)

    # print('\nAgent_0 Data')
    # print(agents['Agent_0'].data)

    # print('\nAll variables in state:')
    # print(list(agents['Agent_0'].data.data_vars.keys()))
   


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
