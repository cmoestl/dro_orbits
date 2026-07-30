#!/usr/bin/env python
# coding: utf-8

# ## DRO Distant retrograde orbits playground
# 
# This code generates distant retrograde orbits (DROs) in the Sun-Earth system depending on the distance from the Sun     
# part of https://github.com/cmoestl/dro_orbits
# 
# For the paper Möstl et al. 2026b ApJ, in prep.
# 
# uses conda environment *dro* (for environment file, see folder env)
#  
# Authors: C. Möstl, Austrian Space Weather Office, GeoSphere Austria    
# https://bsky.app/profile/chrisoutofspace.bsky.social, https://github.com/cmoestl
# 
# last update: July 2026
# 
# ### Issues
# 
# 
# ### Ideas
# - make one table on the r(theta) for the egdes of the cone of acceptance of 5, 7.5, 10, 12.5, 15° - how far behind is the r from rmin? note that this also affects the range of lead times
# - black background plots
# - plotly plot with clickable positions for each ICME event, for homepage
# - check on RLD value for all available ICME events - select events within the 30° domain
# - 3D plot with DROs so one can see the ecliptic and where L4 and L5 are with respect to the solar equator
# - Create DROs around Venus, Mercury or Mars - how do they look like, are would they be useful?
# 

# In[1]:


import time
import os
import requests
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

import pickle
import spiceypy
import pandas as pd
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy import stats
from scipy.signal import argrelextrema
import multiprocessing as mp

import astropy.units as u
from astropy import constants as const

au=const.au.value*1e-3
M_sun = const.M_sun.value  # Sun mass
M_earth = const.M_earth.value   # Earth mass
G=const.G.value*1e-9 # use km
# Sun-Earth system parameters mu: mass parameter (m2/(m1+m2))
mu = M_earth/(M_sun+M_earth)  # Earth mass / (Sun + Earth mass)

# Calculate system parameters
M_total = M_sun + M_earth
omega = np.sqrt(G * M_total / au**3)  # Angular velocity of rotating frame (rad/s) from keplers laws?
T=2*np.pi/omega/86400 #year in decimal days

print('Au in km:',au) # in km
print('M sun',M_sun)
print('M earth',M_earth)
print('G:',G)

print(f"System Parameters:")
print(f"  Mass parameter μ = {mu:.6e}")
print(f"  Earth-Sun distance au = {au:.6e} km")
print(f"  Angular velocity ω = {omega:.6e} rad/s")
print(f"  Orbital period = {2*np.pi/omega/86400:.2f} days\n")


#plotly if needed
#import plotly.graph_objects as go
#from plotly.offline import iplot, init_notebook_mode
#from plotly.subplots import make_subplots
#import plotly.io as pio
#import plotly.express as px
#pio.renderers.default = 'browser'

kernels_path='kernels'

#make sure to convert the current notebook to a script if you want to run it on a server
os.system('jupyter nbconvert --to script dro.ipynb')   


# ### load planetary orbits
# get planetary orbits from spiceypy with files    
# load de442.bsp from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/    
# other files already available in folder: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/    
# saved in folder /kernels
# this is only needed for the Earth here, but more for the animations.ipynb
# 
# 

# In[2]:


#check if de442.bsp is available, otherwise download

def download_if_not_exist(url, filepath):
    if os.path.exists(filepath):
        print(f"File already exists at {filepath}")
        return filepath

    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Download
    response = requests.get(url)
    response.raise_for_status()

    with open(filepath, 'wb') as f:
        f.write(response.content)

    print(f"Downloaded to {filepath}")
    return filepath


filepath='kernels/de442.bsp'
url='https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442.bsp'

download_if_not_exist(url,filepath)

#use two arbitray years for the planets
start=datetime.datetime(2033,1,1)
end=datetime.datetime(2034,1,1)

times = [] 
dt=12 #time resolution for planets is 1 day
# Generate datetimes with increments of dt hours until the end date
current = start
while current <= end:
    times.append(current)
    current += datetime.timedelta(hours=dt)


def cart2sphere_rad(x,y,z):

    r = np.sqrt(x**2+ y**2 + z**2) / au
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi    

    theta=np.deg2rad(theta)
    phi=np.deg2rad(phi)

    return (r, theta, phi)

def generic_furnish(kernels_path):
    generic_path = kernels_path
    generic_kernels = os.listdir(generic_path)
    print(generic_kernels)    
    #spiceypy.furnsh(os.path.join(generic_path, 'de442.bsp'))

    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))

def get_planet_pos(t,kernels_path, planet):
    if spiceypy.ktotal('ALL') < 1:
        generic_furnish(kernels_path)
    pos = spiceypy.spkpos(planet, spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
    r, lat, lon = cart2sphere_rad(pos[0],pos[1],pos[2])
    position = t, pos[0], pos[1], pos[2], r, lat, lon
    return position


def get_planet_positions(time_series,kernels_path,planet):
    positions = []
    for t in time_series:
        position = get_planet_pos(t,kernels_path, planet)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions



def get_planet_pos_hci(t,kernels_path, planet):
    if spiceypy.ktotal('ALL') < 1:
        generic_furnish(kernels_path)
    pos = spiceypy.spkpos(planet, spiceypy.datetime2et(t), "HCI", "NONE", "SUN")[0]
    r, lat, lon = cart2sphere_rad(pos[0],pos[1],pos[2])
    position = t, pos[0], pos[1], pos[2], r, lat, lon
    return position


def get_planet_positions_hci(time_series,kernels_path,planet):
    positions = []
    for t in time_series:
        position = get_planet_pos_hci(t,kernels_path, planet)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions





# make Earth Mercury and Venus positions
generic_furnish(kernels_path)    
print(kernels_path)
mercury=get_planet_positions(times,kernels_path, 'MERCURY_BARYCENTER')
venus=get_planet_positions(times,kernels_path, 'VENUS_BARYCENTER')
earth=get_planet_positions(times,kernels_path, 'EARTH_BARYCENTER')

earth_hci=get_planet_positions_hci(times,kernels_path, 'EARTH_BARYCENTER')



#to matplotlib datenumber
#mercury.time=mdates.date2num(mercury.time)
#venus.time=mdates.date2num(venus.time)
#earth.time=mdates.date2num(earth.time)


sns.set_style('whitegrid')
sns.set_context('paper')   
# Create the plot
fig, ax = plt.subplots(figsize=(8, 3),dpi=100)
plt.plot(mercury.time,mercury.r,'-')
plt.plot(venus.time,venus.r,'-')
plt.plot(earth.time,earth.r,'-')
plt.plot(earth_hci.time,earth_hci.r,'--')

fig, ax = plt.subplots(figsize=(8, 3),dpi=100)
plt.plot(earth_hci.time,np.rad2deg(earth_hci.lon))
plt.plot(earth.time,np.rad2deg(earth.lon))



print('Earth orbit aphelion perihelion',np.min(earth.r), np.max(earth.r))


# ### CR3BP simulation (circular restricted 3 Body Problem)

# equations adapted from https://jan.ucc.nau.edu/~ns46/student/2010/Frnka_2010.pdf

# In[3]:


def cr3bp_equations(t, state):
    #####
    # Equations of motion for the Circular Restricted 3-Body Problem
    # in the rotating reference frame with physical units.    
    # state = [x, y, vx, vy] in km and km/s
    ####
    x, y, vx, vy = state

    # Positions of the primaries in rotating frame
    x1 = -mu * au  # Sun position
    x2 = (1 - mu) * au  # Earth position

    # Distances to the two primaries
    r1 = np.sqrt((x - x1)**2 + y**2)
    r2 = np.sqrt((x - x2)**2 + y**2)

    # Equation 1 in Möstl et al. 2026b ApJ
    ax = (2*omega*vy + omega**2*x - G*M_sun*(x - x1)/r1**3 - G*M_earth*(x - x2)/r2**3)    

    #y2 and y1 is always 0
    ay = (-2*omega*vx + omega**2*y - G*M_sun*y/r1**3 - G*M_earth*y/r2**3)

    return [vx, vy, ax, ay]


def make_dro(initial_state,years):

    days = 366*years  # Simulate for n years
    t_span = (0, days * 86400)      # Time span for integration (in seconds)
    t_eval = np.linspace(t_span[0], t_span[1], days*24) #time resolution is 1 hour

    # Solve the differential equations
    solution = solve_ivp(cr3bp_equations, t_span, initial_state,  t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-8)    

    # Extract trajectory, convert to au
    x = solution.y[0]/au; y = solution.y[1]/au


    return x,y


# ### Numerical simulation of all DROs

# In[29]:


#list for initial conditions for dmin;x and vinit;y
initial_x0_array=[k for k in np.arange(0.74,0.96,0.02)]
print('range of orbits',initial_x0_array[0],initial_x0_array[-1],' # of orbits:',len(initial_x0_array))
#get the initial conditions from the find_dro script

initial_vy_array=[16.822822, 15.41482, 14.012024, 12.65265, 11.306613, 9.983967,     8.6773,   7.394789, 6.1297,       4.8797,3.667334]

#print(initial_x0_array[6])

#time resolution one hour, one year
array_size=366*24



#** add switches for 2 year calculation

#time resolution for 2 years, one hour, use this for the 2 year file *****************
#array_size=2*366*24

#calculate all 11 orbits, and write the orbit solutions in this array
orbits_cart=np.zeros((2,array_size,11))
orbits_polar=np.zeros((2,array_size,11))


t0=time.time()
for i in np.arange(11):
    #print('DRO #',i)
    #print(initial_x0_array[i])
    #print(au)
    #print(initial_vy_array[i])
    #print()

    #initial conditions
    x0 = initial_x0_array[i]*au  # km (between Sun and Earth)
    y0 = 0  # km
    vx0 = 0  # km/s
    vy0 = initial_vy_array[i] # km/s

    orbits_cart[:,:,i]=make_dro([x0, y0, vx0, vy0],1)

    #for 2 years
    #orbits_cart[:,:,i]=make_dro([x0, y0, vx0, vy0],2)### **************

    #polar coordinate conversion, get x and y
    dro_x=orbits_cart[0,:,i] #first coordinate x or y, then data, then orbit #
    dro_y=orbits_cart[1,:,i]
    dro_r= np.sqrt(dro_x**2 + dro_y**2)
    dro_lon = np.arctan2(dro_y, dro_x)

    orbits_polar[0,:,i]=dro_r
    orbits_polar[1,:,i]=dro_lon

print('orbit calculation done, took seconds:',time.time()-t0)

print('access orbits like: x1=orbits_cart[0,:,3], first coordinate x=0 or y=1, then data, then orbit #')

# Perform linear regression using scipy.stats.linregress
slope, intercept, r_value, p_value, std_err = stats.linregress(initial_x0_array, initial_vy_array)

print()
# Print results
print('linear fit')
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Standard error: {std_err:.4f}")
x_fit = np.arange(0.7,1.0,0.01)
y_fit = x_fit*slope + intercept

print()
print('poly fit')
# polynomial fit
coefficients = np.polyfit(initial_x0_array, initial_vy_array, deg=2)
print(f"Coefficients (highest to lowest degree): {coefficients}")
# Create polynomial function from coefficients
poly_func = np.poly1d(coefficients)

# for plot
y_fit_poly = poly_func(x_fit)

#plot only results for the initial conditions for checking
plt.plot(initial_x0_array,initial_vy_array,'ko', linestyle='--',linewidth=1)


##### ************
## write one file with all DRO orbits in cartesian and polar coordinates for 2 years for the animations
#filename_pickle='dro_all_2years.p'
#with open(file_dir+filename_pickle, 'wb') as f:
#        pickle.dump(orbits_polar, f)
#        pickle.dump(orbits_cart, f)



# ## Figure 1 initial conditions and DRO solutions in cartesian coordinates

# In[30]:


sns.set_style('whitegrid')
sns.set_context('paper')   
# Create the plot

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15, 8))

#orbits

dro_colors = [
    'red',
    'orangered',
    'orange',
    'gold',
    'yellow',
    'yellowgreen',
    'green',
    'teal',
    'blue',
    'indigo',
    'violet'
]


#main 4 0.74, 0.80, 0.86, 0.92
ax2.plot(orbits_cart[0,:,0], orbits_cart[1,:,0], dro_colors[0], linewidth=1.5, alpha=1.0, label='DRO 1 0.74 au')
ax2.plot(orbits_cart[0,:,1], orbits_cart[1,:,1], dro_colors[1], linewidth=1.5, alpha=1.0, label='DRO 2 0.76 au')
ax2.plot(orbits_cart[0,:,2], orbits_cart[1,:,2], dro_colors[2], linewidth=1.5, alpha=1.0, label='DRO 3 0.78 au')

ax2.plot(orbits_cart[0,:,3], orbits_cart[1,:,3], dro_colors[3], linewidth=1.5, alpha=1.0, label='DRO 4 0.80 au')
ax2.plot(orbits_cart[0,:,4], orbits_cart[1,:,4], dro_colors[4], linewidth=1.5, alpha=1.0, label='DRO 5 0.82 au')
ax2.plot(orbits_cart[0,:,5], orbits_cart[1,:,5], dro_colors[5], linewidth=1.5, alpha=1.0, label='DRO 6 0.84 au')

ax2.plot(orbits_cart[0,:,6], orbits_cart[1,:,6], dro_colors[6], linewidth=1.5, alpha=1.0, label='DRO 7 0.86 au SHIELD', linestyle='--')
ax2.plot(orbits_cart[0,:,7], orbits_cart[1,:,7], dro_colors[7], linewidth=1.5, alpha=1.0, label='DRO 8 0.88 au')
ax2.plot(orbits_cart[0,:,8], orbits_cart[1,:,8], dro_colors[8], linewidth=1.5, alpha=1.0, label='DRO 9 0.90 au')

ax2.plot(orbits_cart[0,:,9], orbits_cart[1,:,9], dro_colors[9], linewidth=1.5, alpha=1.0, label='DRO 10 0.92 au HENON',linestyle='--')
ax2.plot(orbits_cart[0,:,10], orbits_cart[1,:,10], dro_colors[10], linewidth=1.5, alpha=1.0, label='DRO 11 0.94 au')

# Plot Sun - fixed at origin shifted by -mu
sun_x = -mu  
# ax.scatter(0,0,s=100,c='yellow',alpha=1, edgecolors='black', linewidth=0.3, label='Sun')
# Plot Earth - fixed at (1-mu)
earth_x = (1 - mu) 
ax2.plot(earth_x, 0, 'o', color='blue', markersize=5, label='Earth', zorder=5)

# Add distance circles for reference
circle1 = plt.Circle((sun_x, 0), 1, fill=True, color='gray', linestyle='--', alpha=0.2, label='< 1 au')
circle2 = plt.Circle((sun_x, 0), 0.7282, fill=True, color='gold', linestyle='-', alpha=0.2, label='< Venus aphelion')

ax2.add_patch(circle1)
ax2.add_patch(circle2)
ax2.set_xlabel('x [au]', fontsize=15)
ax2.set_ylabel('y [au]', fontsize=15)
#ax.set_title(f'Circular Restricted 3-Body Problem: Sun-Earth System\n(Rotating Reference Frame, {days} days simulation)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8)
ax2.xaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.grid(True, alpha=1.0, linestyle='-')
ax2.set_xlim(0.5, 1.6)
ax2.set_ylim(-0.6, 0.6)
ax2.tick_params(axis='x', labelsize=15) 
ax2.tick_params(axis='y', labelsize=15) 
ax2.set_aspect('equal')


###############add panel for initial conditions
########### dependence of initial vy peed on heliocentric distance

for i in np.arange(11):
    ax1.scatter(initial_x0_array[i], initial_vy_array[i], facecolor=dro_colors[i], s=200,marker='o', edgecolor='black',zorder=3 )

ax1.plot(x_fit, y_fit, label='linear fit', color='dimgrey')
ax1.plot(x_fit, y_fit_poly, label='polynomial 2nd degree fit', color='dimgrey',linestyle='--')

ax1.legend(loc='upper right', fontsize=15)
ax1.set_xlabel('$min(x)$ [au]',fontsize=15)

ax1.set_ylabel('$v_{y;init}$ [km s$^{-1}]$',fontsize=15)
ax1.set_xlim(0.7, 1.0)
ax1.set_ylim(0,20)
ax1.tick_params(axis='x', labelsize=15) 
ax1.tick_params(axis='y', labelsize=15) 

plt.tight_layout()

### labels
fig.text(0.01, 0.98, '(a)', fontsize=18, va='top', ha='left')
fig.text(0.50, 0.98, '(b)', fontsize=18, va='top', ha='left')

plt.savefig('results/fig1_initial_cartesian_dro.png', dpi=300,bbox_inches='tight')
plt.savefig('results/fig1_initial_cartesian_dro.pdf', dpi=300,bbox_inches='tight')


# ## write orbits in pickle and txt files 
# 

# In[31]:


file_dir='orbit_files/'


years=1
days = 366*years  # Simulate for 1 year with 366 days 
t_span = (0, days * 24)      # Time span in hours 
t_eval = np.linspace(t_span[0], t_span[1]-1, days*24) #time resolution is 1 hour

print(t_eval[-10:-1])

for i in np.arange(11):

    #access orbits like: x1=orbits_cart[0,:,3], first coordinate x=0 or y=1, then data, then orbit #    
    #first cartesian x y, and then r and longitude in rad, in recarray
    dro_i = np.rec.fromarrays([t_eval,orbits_cart[0,:,i], orbits_cart[1,:,i], orbits_polar[0,:,i], orbits_polar[1,:,i]],dtype=[('time', 'f8'),('x', 'f8'), ('y', 'f8'), ('r', 'f8'), ('lon', 'f8')])

    #define filename        
    dmin_str='0_'+str(f"{orbits_cart[0,0,i]:.2f}")[2:4]

    # write pickle files    
    filename_pickle='dro'+str(f"{i:02d}")+'__'+dmin_str+'au.p'
    with open(file_dir+filename_pickle, 'wb') as f:
        pickle.dump(dro_i, f)

    #write ASCII files 
    filename_txt='dro'+str(f"{i:02d}")+'__'+dmin_str+'au.txt'
    np.savetxt(file_dir+filename_txt, dro_i, header='time [hours] x [au] y [au] r [au] lon [rad] HEE coordinates', fmt='%.6f')

    print('orbit written into', file_dir+filename_pickle, ' ',filename_txt, ' ',dmin_str)


## write one file with all DRO orbits in cartesian and polar coordinates
filename_pickle='dro_all.p'
with open(file_dir+filename_pickle, 'wb') as f:
        pickle.dump(orbits_polar, f)
        pickle.dump(orbits_cart, f)


############### Example for reading the pickle files
filename='orbit_files/dro07__0_88au.p'
with open(filename,'rb') as f:
    dro = pickle.load(f)

fig, ax = plt.subplots(1,1,figsize=(3, 3))
ax.plot(dro.x,dro.y)
ax.set_aspect('equal')    


# ## Figure 2 DRO and planets plot, spacecraft distribution
# 

# In[32]:


sns.set_style('whitegrid')
sns.set_context('talk')    

fig, ax = plt.subplots(1,figsize=(10, 8),subplot_kw={'projection': 'polar'},dpi=100)    

#ax.text(0,0,'Sun', color='black', ha='center',fontsize=fsize-5,verticalalignment='top')
#ax.text(0,1.2,'Earth', color='mediumseagreen', ha='center',fontsize=fsize-5,verticalalignment='center')
symsize_planet=5
# Sun
ax.scatter(0,0,s=100,c='yellow',alpha=1, edgecolors='black', linewidth=0.3)
ax.scatter(earth.lon, earth.r, s=symsize_planet, c='mediumseagreen', alpha=1,lw=0,zorder=3,marker=None, label='Earth')  

#Sun-Earth line
ax.plot(np.zeros(11),np.arange(0,1.1,0.1),c='k',lw=1,alpha=0.8,linestyle='-',zorder=0)

#1 au circle
ax.plot(np.deg2rad(np.arange(0,360)),np.zeros(360)+1,lw=1,alpha=0.5,linestyle='-',c='black', marker=None)


ax.plot(orbits_polar[1,:,0], orbits_polar[0,:,0], dro_colors[0], linewidth=1.5, alpha=1.0, label='DRO 1 0.74 au')
ax.plot(orbits_polar[1,:,3], orbits_polar[0,:,3], dro_colors[3], linewidth=1.5, alpha=1.0, label='DRO 4 0.80 au')
ax.plot(orbits_polar[1,:,6], orbits_polar[0,:,6], dro_colors[6], linewidth=1.5, alpha=1.0, label='DRO 7 0.86 au SHIELD')
ax.plot(orbits_polar[1,:,9], orbits_polar[0,:,9], dro_colors[9], linewidth=1.5, alpha=1.0, label='DRO 10 0.92 au HENON')

# grid line appearance
ax.grid(True, color='gray', linewidth=0.5, linestyle='--', alpha=0.6)

#shaded areas for planets
inner_venus=0.718440 
outer_venus=0.728213
theta = np.linspace(0, 2 * np.pi, 200)
r_inner = np.full_like(theta, inner_venus)
r_outer = np.full_like(theta, outer_venus)
ax.fill_between(theta, r_inner, r_outer, color='gold', alpha=0.7, label='Venus')

inner_mercury=   0.307499      
outer_mercury= 0.466697          
r_inner = np.full_like(theta, inner_mercury)
r_outer = np.full_like(theta, outer_mercury)
ax.fill_between(theta, r_inner, r_outer, color='grey', alpha=0.4, label='Mercury')

fsize=15
#set axes
ax.set_theta_zero_location('E')
plt.rgrids((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.6,2.0,2.5),('0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.1','1.2','1.3','1.6 AU','2.0','2.5'),angle=35, fontsize=fsize-5,alpha=0.4)

degrees = np.arange(0, 360, 10)
ax.set_xticks(np.radians(degrees))

deg_label = np.concatenate([np.arange(0, 190, 10), np.arange(-170,0,10)])
ax.set_xticklabels([f'{d}°' for d in deg_label], fontsize=10)
#ax.legend(bbox_to_anchor=(0.95, 1.03), loc='upper left',fontsize=10)
ax.legend(bbox_to_anchor=(0.05, 0.65), loc='upper left',fontsize=10, framealpha=1.0)
ax.set_ylim(0, 1.3) 

################### plot spacecraft
#number of SHIELD spacecraft
nr_sc=9
print(array_size) # all time datapoints 
interval=int(np.round(array_size/nr_sc)) #round to nearest hour
#indices of shield spacecraft equidistant in time over 1 year
shield_i=np.arange(0,array_size,interval)[0:9] 

print('Number of SHIELD Spacecraft:',nr_sc)
print('Interval in days:',interval/24)
print('longitudes:',np.round(np.rad2deg(orbits_polar[1,shield_i,0])))

ax.scatter(orbits_polar[1,shield_i,0], orbits_polar[0,shield_i,0],marker='o', c=dro_colors[0], alpha=1.0,s=15)
ax.scatter(orbits_polar[1,shield_i,3], orbits_polar[0,shield_i,3],marker='o', c=dro_colors[3], alpha=1.0,s=15)
ax.scatter(orbits_polar[1,shield_i,6], orbits_polar[0,shield_i,6],marker='o', c=dro_colors[6], alpha=1.0,s=15)
ax.scatter(orbits_polar[1,shield_i,9], orbits_polar[0,shield_i,9],marker='o', c=dro_colors[9], alpha=1.0,s=15)

plt.tight_layout()
plt.savefig('results/fig2_polar.png', dpi=300,bbox_inches='tight')
plt.savefig('results/fig2_polar.pdf', dpi=300,bbox_inches='tight')


# ## Figure 3 plots for DRO characteristics

# In[33]:


####### relationship between minimum distance and widest point in y in au 

sns.set_style('whitegrid')
sns.set_context('talk')    

fig = plt.figure(figsize=(15, 12))

gs = gridspec.GridSpec(2, 4, figure=fig)

ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4])
ax3 = fig.add_subplot(gs[1, 1:3])  # centered, half-width

########################## (a) widest extension in dy
minx_list=[]
maxy_list=[]

for i in np.arange(11):

    #min of x component, all data, dro with number i
    minx=np.min(orbits_cart[0,:,i])
    # max of y component, all data, dro with number i
    maxy=np.max(orbits_cart[1,:,i])

    #make lists for later usage
    minx_list.append(minx)
    maxy_list.append(maxy)

    ax1.scatter(minx,maxy, marker='o',c=dro_colors[i],edgecolor='black',zorder=3)


ax1.set_xlabel('$min(x)$ [au]',fontsize=15)
ax1.set_ylabel('$max(y)$ [au]',fontsize=15)
ax1.set_xlim(0.7, 1.0)
ax1.set_ylim(0,0.6)
ax1.tick_params(axis='x', labelsize=15) 
ax1.tick_params(axis='y', labelsize=15) 
ax1.xaxis.set_major_locator(MultipleLocator(0.04))



# Perform linear regression using scipy.stats.linregress
slope, intercept, r_value, p_value, std_err = stats.linregress(minx_list, maxy_list)

print('fit for dxmin vs dymax')
# Print results
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Standard error: {std_err:.4f}")
print()

y_fit1 = x_fit*slope + intercept
ax1.plot(x_fit,y_fit1, c='dimgrey',alpha=0.8,zorder=1)

print(' Factor for widest extension in y compared to distance from Earth: ',np.mean(np.array(maxy_list)/(1-np.array(minx_list))))

#rev_x_fit=1-x_fit
#print(np.round(rev_x_fit,2))

########################## (b) widest extension in longitude

maxlong_list=[]

for i in np.arange(11):

    # max of longitude in polar array, all data, dro with number i
    maxlong=np.rad2deg(np.max(orbits_polar[1,:,i]))
    maxlong_list.append(maxlong)    
    ax2.scatter(minx_list[i],maxlong, marker='o',c=dro_colors[i],edgecolor='black',zorder=3)

ax2.set_xlabel('$r_{min}$ [au]',fontsize=15)
ax2.set_ylabel(r'$max( \theta )$ [°]')
ax2.set_xlim(0.7, 1.0)
ax2.set_ylim(0,35)
ax2.tick_params(axis='x', labelsize=15) 
ax2.tick_params(axis='y', labelsize=15) 
ax2.yaxis.set_major_locator(MultipleLocator(2))
ax2.xaxis.set_major_locator(MultipleLocator(0.04))

# Perform linear regression using scipy.stats.linregress
slope, intercept, r_value, p_value, std_err = stats.linregress(minx_list, maxlong_list)

# Print results
print('fit for dxmin vs theta')
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Standard error: {std_err:.4f}")

y_fit2 = x_fit*slope + intercept
ax2.plot(x_fit,y_fit2, c='dimgrey')


# Rectangle((x_start, y_start), width, height)

box = Rectangle(((30-intercept)/slope, 0), 0.15, 30, 
                facecolor='yellow', 
                alpha=0.2,
                edgecolor='k',
                linewidth=2, label='< 30° longitude ')
ax2.add_patch(box)

box = Rectangle(((25-intercept)/slope, 0), 0.15, 25, 
                facecolor='greenyellow', 
                alpha=0.2,
                edgecolor='k',
                linewidth=2, label='< 25° longitude ')
ax2.add_patch(box)



box = Rectangle(((20-intercept)/slope, 0), 0.175, 20, 
                facecolor='orange', 
                alpha=0.2,
                edgecolor='k',
                linewidth=2, label='< 20° longitude ')
ax2.add_patch(box)


# Rectangle((x_start, y_start), width, height)
box = Rectangle(((15-intercept)/slope, 0), 0.15, 15, 
                facecolor='red', 
                alpha=0.2,
                edgecolor='k',
                linewidth=2, label='< 15° longitude ')
ax2.add_patch(box)


box = Rectangle(((10-intercept)/slope, 0), 0.15, 10, 
                facecolor='purple', 
                alpha=0.2,
                edgecolor='k',
                linewidth=2, label='< 10° longitude ')
ax2.add_patch(box)


print()
print('For Table 1:')
print('for a given maximum extension in longitude, what is the dmin of the DRO?')

for i in np.arange(5,35,1):
    print('max longitude is ',i,' degree, then dmin is ',(i-intercept)/slope)

ax2.legend()


########################## (c) orbital periods

#time axis same as for sim above in the function
years=1
days = 366*years  # Simulate for 1 year
t_span = (0, days * 24)      # Time span in hours
t_eval = np.linspace(t_span[0], t_span[1]-1, days*24) #time resolution is 1 hour

labeling=['0.74 au','0.76 au','0.78 au','0.80 au','0.82 au','0.84 au','0.86 au','0.88 au','0.90 au','0.92 au','0.94 au']

for i in np.arange(11):
    ax3.plot(t_eval/(24),np.rad2deg(orbits_polar[1,:,i]),c=dro_colors[i], label=labeling[i])

ax3.set_xlabel('days')
ax3.set_ylabel(r'longitude $\theta$ [°]')
ax3.set_xlim(0, 365)
ax3.legend(fontsize=10)

#with SHIELD distribution
#ax.scatter(x1[shield_i],y1[shield_i],marker='o',c='black')
#ax.scatter(x2[shield_i],y2[shield_i],marker='o',c='red')
#ax.scatter(x3[shield_i],y3[shield_i],marker='o',c='blue')
#ax.scatter(x4[shield_i],y4[shield_i],marker='o',c='green')
#ax.scatter(x5[shield_i],y5[shield_i],marker='o',c='orange')

############
### labels
fig.text(0.01, 0.98, '(a)', fontsize=20, va='top', ha='left')
fig.text(0.50, 0.98, '(b)', fontsize=20, va='top', ha='left')
fig.text(0.25, 0.48, '(c)', fontsize=20, va='top', ha='left')

plt.tight_layout()
plt.savefig('results/fig3_characterize.pdf', dpi=300,bbox_inches='tight')
plt.savefig('results/fig3_characterize.png', dpi=300,bbox_inches='tight')


# ### Table for cone of acceptance r(theta_c)

# In[34]:


#make one table on the r(theta) for the cone of acceptance of 5, 7.5, 10, 12.5, 15° - how far behind is the r from rmin?

labeling=['0.74 au','0.76 au','0.78 au','0.80 au','0.82 au','0.84 au','0.86 au','0.88 au','0.90 au','0.92 au','0.94 au']

#cone of acceptance is 5 to 20° in 1° steps
conedeg=np.arange(5,21,2.5)
cone=np.deg2rad(np.arange(5,21,2.5))
print('cone of acceptance',np.rad2deg(cone))

print()

#go through all orbits
for i in np.arange(11):
    #print(labeling[i])
    #theta
    #orbits_polar[1,:,i]
    #r    
    print('rmin = r(theta=0)',np.min(np.round(orbits_polar[0,:,i],3)))

    for conej in cone:

        #check whether this dro extends over the cone of acceptance
        if conej < np.max(orbits_polar[1,:,i]):
            coneindex=np.where(orbits_polar[1,:,i] > conej)[0][0]
            #print(coneindex)
            #check for index with theta - should be the cone value
            #print(np.rad2deg(orbits_polar[1,coneindex,i]))
            print('theta_cone = ',np.round(np.rad2deg(conej),1),'deg, r(theta_cone) =', np.round(orbits_polar[0,coneindex,i],4))



    print()



# ## Figure 4  lead times

# In[35]:


##analysis of distance vs lead time of different types of CMEs, assuming radial propagating front

speed=400 #km/s
leadmax=(1.0-0.7)*au/speed/(3600)
#print(f'lead time for 400 km/s wind for 0.7 au is {leadmax:.2f} hours')
#print('This is the maximum plot range, so it includes Venus')


############### plot
sns.set_style('whitegrid')
sns.set_context('talk')    

fig, ax = plt.subplots(1,figsize=(12, 6),dpi=100)   

#colors = ['red', 'orangered', 'gold', 'limegreen', 'dodgerblue', 'darkviolet', 'purple']

style = [
    'solid',
    'dashed',
    'dotted',
    'dashdot',
    (0, (3, 1, 1, 1)),   # densely dashdotted
    (0, (5, 5)),         # custom dashed pattern
    (0, (1, 1))          # densely dotted
]

# Draw dashed vertical lines at each location, and L1
for i in np.arange(11):
    ax.axvline(x=initial_x0_array[i], c=dro_colors[i], linestyle='-', linewidth=2)
#L1
ax.axvline(x=0.99, c='k', linestyle='-', linewidth=2)



k=0
for i in [400,600,800,1000,1500,2000,2500]:    
    speed=i #km/s
    leaddist=np.linspace(0.7,1.0,16)
    leadtime=(1.0-leaddist)*au/speed/(3600)

    ax.plot(leaddist,leadtime,label=f'{i} km s$^{{-1}}$',color='k', linestyle=style[k])
    k=k+1

ax.legend(framealpha=1.0)    
ax.set_ylabel('Lead time [hours]')
ax.set_xlabel('DRO minimum heliocentric distance $r_{min}$ [au]') 
ax.set_ylim(0,32)
ax.set_yticks(np.arange(0,35,2))
ax.set_xticks(np.arange(0.7,1.0,0.025))

ax.xaxis.set_major_locator(MultipleLocator(0.02))
ax.yaxis.set_major_locator(MultipleLocator(3))

ax.grid(True, color='gray', linewidth=0.5, linestyle='-', alpha=0.6)

plt.tight_layout()

plt.savefig(f'results/fig4_lead_time.png', dpi=300,bbox_inches='tight')
plt.savefig(f'results/fig4_lead_time.pdf', dpi=300,bbox_inches='tight')


print('Table for lead time results')

k=0


for i in [400,600,800,1000,1500,2000,2500]:    
    speed=i #km/s
    dxmin=np.linspace(0.74,0.94,11)
    leadtime=(1.0-dxmin)*au/speed/(3600)

    print()
    print(speed,'km/s')
    for k in np.arange(0,11):
        print(f"{np.round(dxmin[k],2):01f}",'au', np.round(leadtime[k],1),'hours')


# ## Figure 5  gap analysis 

# get intervals for 3,6,9,12 spacecraft

# In[36]:


# number of spacecraft
nr_sc=np.array([3,4,5,6,7,8,9,10,11,12,13,14,15,16])

print(array_size) # all time datapoints 
interval=np.int16(array_size/nr_sc) #integer for using as index

#indices of shield spacecraft equidistant distributed in time over 366 days, this is relevant for all orbits
sc_int3=np.arange(0,array_size,interval[0]) 
sc_int4=np.arange(0,array_size,interval[1]) 
sc_int5=np.arange(0,array_size,interval[2]) 
sc_int6=np.arange(0,array_size,interval[3])
sc_int7=np.arange(0,array_size,interval[4]) 
sc_int8=np.arange(0,array_size,interval[5]) 
sc_int9=np.arange(0,array_size,interval[6]) 
sc_int10=np.arange(0,array_size,interval[7]) 
sc_int11=np.arange(0,array_size,interval[8]) 
sc_int12=np.arange(0,array_size,interval[9]) 
sc_int13=np.arange(0,array_size,interval[10]) 
sc_int14=np.arange(0,array_size,interval[11]) 
sc_int15=np.arange(0,array_size,interval[12]) 
sc_int16=np.arange(0,array_size,interval[13]) 

print(sc_int3)
print(sc_int6)
print(sc_int8)
print(sc_int9)
print(sc_int12)


print()
## case 9 s/c
print('example')
print('Number of SHIELD Spacecraft:',nr_sc[3])
print('Interval in days (rounded):',np.round(sc_int9/24))
#print('longitudes:',np.round(np.rad2deg(orbits_polar[1,shield_i,0])))
#indices of each spacecraft at start
print(sc_int9)


#####################
#orbits to focus on 0 3 6 9 for plot
#t_eval is time

#longitude of each spacecraft
dlon=np.rad2deg(orbits_polar[1,:,6]) #longitude, orbit #6

## then advance by n hours 
days=20
n=24*days
print(sc_int9+n)
sc_int9n=sc_int9+n

fig, ax = plt.subplots(1,figsize=(5, 5),dpi=100)   
ax.plot(t_eval/24,dlon)
ax.plot(t_eval[sc_int9n]/24, dlon[sc_int9n],marker='o', linestyle='None')
ax.set_ylabel('longitude [°]')
ax.set_xlabel('time [days]')


# ### get delta to Sun-Earth line in heliospheric longitude

# In[37]:


####### example for orbit with 9 sc ###########
#days=19 #step forward
#n=np.int16(24*days) #round to nearest hour
#print(np.round(dlon[sc_int9+n],1))
#####################################


############### go through all 11 orbits and use the given number of spacecraft, 
#answering: what is the maximum separation of the spacecraft to the Sun-Earth line in HEE longitude?
#check when first spacecraft is further away in longitude from the SE line than the last spacecraft

def get_delta(sc_int_array):
    #for each dmin orbit distance write out the index with the maximum separation of the first sc before the last sc is closer
    delta_index=np.zeros(np.size(initial_x0_array))
    delta_value=np.zeros(np.size(initial_x0_array))    

    #going through all 11 different orbits, for a given number of spacecraft set by sc_int_array
    for i in np.arange(np.size(initial_x0_array)):

        hours=0
        #get orbit longitude values: 1 is longitude, : are all longitude values, i is the orbit (defined by dmin)
        dlon=np.rad2deg(orbits_polar[1,:,i])

        # use number of spacecraft you want to analyse, sc_int9
        # starting values
        first_sc=np.abs(dlon[sc_int_array+hours][0])
        last_sc=np.abs(dlon[sc_int_array+hours][-1])
        #print(first_sc, last_sc)

        #find index when both spacecraft are at maximum separation to Sun-Earth line
        while first_sc < last_sc:            
            #longitude of first spacecraft along the DRO, starting at the Sun-Earth line and going towards west, along the orbit
            first_sc=np.abs(dlon[sc_int_array+hours][0])
            #longitude of last spacecraft along the orbit, absolute value so comparable with first spacecraft
            last_sc=np.abs(dlon[sc_int_array+hours][-1])
            #print(first_sc, last_sc)
            delta_index[i]=hours
            hours=hours+1

        #print(delta_index)
        delta_value[i]=np.rad2deg(orbits_polar[1,int(delta_index[i]),i])

    return delta_value

#alternative version - take difference between s/c in hours and divide by 2, and get this as the 
#index of maximum separation to the Sun-Earth line

def get_delta2(sc_int_array):

    delta_index=np.zeros(np.size(initial_x0_array))
    delta_value=np.zeros(np.size(initial_x0_array))   

    for i in np.arange(np.size(initial_x0_array)):
        #take point symmetric around sun-earth line, and divide by 2 
        delta_index[i]=sc_int_array[1]/2
        #print(delta_index)

        delta_value[i]=np.rad2deg(orbits_polar[1,int(delta_index[i]),i])

    #print(delta_value)    


    return delta_value



#method 1
delta_value3=get_delta(sc_int3)
delta_value6=get_delta(sc_int6)
delta_value9=get_delta(sc_int9)
delta_value12=get_delta(sc_int12)

#method 2 further used
delta2_value3=get_delta2(sc_int3)
delta2_value4=get_delta2(sc_int4)
delta2_value5=get_delta2(sc_int5)
delta2_value6=get_delta2(sc_int6)
delta2_value7=get_delta2(sc_int7)
delta2_value8=get_delta2(sc_int8)
delta2_value9=get_delta2(sc_int9)
delta2_value10=get_delta2(sc_int10)
delta2_value11=get_delta2(sc_int11)
delta2_value12=get_delta2(sc_int12)
delta2_value13=get_delta2(sc_int13)
delta2_value14=get_delta2(sc_int14)
delta2_value15=get_delta2(sc_int15)
delta2_value16=get_delta2(sc_int16)


print('dxmin:',np.round(initial_x0_array,2))
print('delta for 4 spacecraft:',delta2_value4)
print('delta for 6 spacecraft:',delta2_value6)
print('delta for 9 spacecraft:',delta2_value9)
print('delta for 12 spacecraft:',delta2_value12)
print('delta for 15 spacecraft:',delta2_value15)


#put all in one array, these are 10 different spacecraft numbers
#delta2_all=[delta2_value3,delta2_value4,delta2_value5,delta2_value6,delta2_value7,
#            delta2_value8,delta2_value9,delta2_value10,delta2_value11,delta2_value12]

#e.g. get all values for 3 spacecraft
#delta2_all[0]


#not fully consistent, need to check method 1 (numerical errors?)
#print(delta_value3)
#print(delta2_value3)


# In[38]:


####### plot results from gap analysis

#furthermore, method 2 is used

############## plot

sns.set_style('whitegrid')
sns.set_context('talk')    

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='polar')


#plot for each orbit with dmin the longitudes of the spacecraft in HEE, absolute difference to Sun-Earth line
#ax1.scatter(initial_x0_array,delta_value3,c=dro_colors, marker='v', label='3 spacecraft',edgecolor='k')
#ax1.scatter(initial_x0_array,delta_value6,c=dro_colors, marker='s', label='6 spacecraft',edgecolor='k')
#ax1.scatter(initial_x0_array,delta_value9,c=dro_colors, marker='D', label='9 spacecraft',edgecolor='k')
#ax1.scatter(initial_x0_array,delta_value12,c=dro_colors, marker='o', label='12 spacecraft',edgecolor='k')

ms=60
ax1.scatter(initial_x0_array,delta2_value4,c=dro_colors, marker='v', label='4 spacecraft',edgecolor='k',zorder=3,s=ms)
ax1.scatter(initial_x0_array,delta2_value6,c=dro_colors, marker='s', label='6 spacecraft',edgecolor='k',zorder=3,s=ms)
ax1.scatter(initial_x0_array,delta2_value9,c=dro_colors, marker='D', label='9 spacecraft',edgecolor='k',zorder=3,s=ms)
ax1.scatter(initial_x0_array,delta2_value12,c=dro_colors, marker='o', label='12 spacecraft',edgecolor='k',zorder=3,s=ms)

print()
print('poly fits')

coefficients = np.polyfit(initial_x0_array, delta2_value4, deg=2)
print(f"4 spacecraft poly fit with 2nd degree: {coefficients}")
# Create polynomial function from coefficients
poly_func = np.poly1d(coefficients)
x_fit = np.arange(0.7,1.0,0.01)
y_fit_poly = poly_func(x_fit)
ax1.plot(x_fit, y_fit_poly,'k', linestyle='--',linewidth=1)

# polynomial fits
coefficients = np.polyfit(initial_x0_array, delta2_value6, deg=2)
print(f"6 spacecraft poly fit with 2nd degree: {coefficients}")
# Create polynomial function from coefficients
poly_func = np.poly1d(coefficients)
x_fit = np.arange(0.7,1.0,0.01)
y_fit_poly = poly_func(x_fit)
ax1.plot(x_fit, y_fit_poly,'k', linestyle='--',linewidth=1)

# polynomial fits
coefficients = np.polyfit(initial_x0_array, delta2_value9, deg=2)
print(f"9 spacecraft poly fit with 2nd degree: {coefficients}")
# Create polynomial function from coefficients
poly_func = np.poly1d(coefficients)
x_fit = np.arange(0.7,1.0,0.01)
y_fit_poly = poly_func(x_fit)
ax1.plot(x_fit, y_fit_poly,'k', linestyle='--',linewidth=1)

# polynomial fits
coefficients = np.polyfit(initial_x0_array, delta2_value12, deg=2)
print(f"12 spacecraft poly fit with 2nd degree: {coefficients}")
# Create polynomial function from coefficients
poly_func = np.poly1d(coefficients)
x_fit = np.arange(0.7,1.0,0.01)
y_fit_poly = poly_func(x_fit)
ax1.plot(x_fit, y_fit_poly,'k', linestyle='--',linewidth=1)


ax1.legend()
ax1.set_ylabel(r'$ max(\Delta_{SE})$ [°] longitude, HEE')
ax1.set_xlabel(r'DRO minimum heliocentric distance $r_{min}$ [au]') 
ax1.xaxis.set_major_locator(MultipleLocator(0.02))
ax1.yaxis.set_major_locator(MultipleLocator(2))
ax1.set_ylim(0,30)
ax1.set_xlim(0.72,0.99)
ax1.grid(True, color='gray', linewidth=0.8, linestyle='-', alpha=0.6)


ax1.xaxis.set_minor_locator(MultipleLocator(0.01))
ax1.yaxis.set_minor_locator(MultipleLocator(1))
# Enable minor gridlines
ax1.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.8)

#domain with 10°
ax1.axhspan(ymin=-1, ymax=15, color='yellowgreen', alpha=0.3, zorder=0)
ax1.axhspan(ymin=-1, ymax=10, color='green', alpha=0.2, zorder=0)
ax1.axhspan(ymin=-1, ymax=5, color='green', alpha=0.3, zorder=0)



############################################## DROs for context

fsize=15
symsize_planet=10
ax2.scatter(earth.lon, earth.r, s=symsize_planet, c='mediumseagreen', alpha=1,lw=0,zorder=3,marker=None, label='Earth')  
ax2.plot(orbits_polar[1,:,0], orbits_polar[0,:,0], dro_colors[0], linewidth=1.5, alpha=1.0, label='DRO 1 0.74 au')
ax2.plot(orbits_polar[1,:,3], orbits_polar[0,:,3], dro_colors[3], linewidth=1.5, alpha=1.0, label='DRO 4 0.80 au')
ax2.plot(orbits_polar[1,:,6], orbits_polar[0,:,6], dro_colors[6], linewidth=1.5, alpha=1.0, label='DRO 7 0.86 au SHIELD')
ax2.plot(orbits_polar[1,:,9], orbits_polar[0,:,9], dro_colors[9], linewidth=1.5, alpha=1.0, label='DRO 10 0.92 au HENON')


#spacecraft distributed
shift=475
scsize=25
ax2.scatter(orbits_polar[1,shield_i+shift,0], orbits_polar[0,shield_i+shift,0],marker='o', c=dro_colors[0], alpha=1.0,s=scsize)
ax2.scatter(orbits_polar[1,shield_i+shift,3], orbits_polar[0,shield_i+shift,3],marker='o', c=dro_colors[3], alpha=1.0,s=scsize)
ax2.scatter(orbits_polar[1,shield_i+shift,6], orbits_polar[0,shield_i+shift,6],marker='o', c=dro_colors[6], alpha=1.0,s=scsize)
ax2.scatter(orbits_polar[1,shield_i+shift,9], orbits_polar[0,shield_i+shift,9],marker='o', c=dro_colors[9], alpha=1.0,s=scsize)

#sun earth line and 1 au circle
ax2.plot(np.deg2rad(np.arange(0,360)),np.zeros(360)+1,lw=1,alpha=0.8,linestyle='--',c='black', marker=None)
ax2.plot(np.zeros(5),np.arange(0.7,1.1,0.1),c='k',lw=1,alpha=0.8,linestyle='--')

ax2.set_theta_zero_location('E')
ax2.set_thetamin(35)      # Start angle in degrees
ax2.set_thetamax(-35)
# cutout in r
ax2.set_rmin(0.70)
ax2.set_rmax(1.30)
ax2.set_rorigin(0)

degrees = np.arange(-35,35,5)
ax2.set_xticks(np.radians(degrees))
ax2.set_xticklabels([f'{d}°' for d in degrees], fontsize=15)

ax2.legend(loc=3,fontsize=12)

fig.text(0.55, 0.33, '9 spacecraft case', fontsize=14, va='top', ha='left')

fig.text(0.01, 0.98, '(a)', fontsize=22, va='top', ha='left')
fig.text(0.55, 0.98, '(b)', fontsize=22, va='top', ha='left')


plt.tight_layout()

plt.savefig(f'results/fig5_gap_analysis.png', dpi=300,bbox_inches='tight')
plt.savefig(f'results/fig5_gap_analysis.pdf', dpi=300,bbox_inches='tight')


# ### Figure 6 heatmap for gap analysis
# 

# In[39]:


### add analysis for heat map how many spacecraft are needed
#delta2_value3

#get variables again for x and y 

labeling=['0.74 au','0.76 au','0.78 au','0.80 au','0.82 au','0.84 au','0.86 au','0.88 au','0.90 au','0.92 au','0.94 au']
labeling_clean = ['0.74', '0.76', '0.78', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94']


#yaxis for table
initial_x0_array

#xaxis for table is cone of acceptance is 5 to 20° in 1° steps
conedeg=np.arange(4,22,2)
cone=np.deg2rad(np.arange(4,22,2))

# define a heatmap for all combinations of orbits and cone of acceptance
heatmap=np.zeros([len(initial_x0_array),len(cone)])


print('------')
print('cone of acceptance',np.rad2deg(cone))
print('dros',np.round(initial_x0_array,2))
print('sc numbers',nr_sc)
print('------')


print()
print()


#get number of spacecraft that fulfills the criteria that for all times a spacecraft is within the cone of acceptance

#go through each orbit
for i in np.arange(len(initial_x0_array)):    

    #go through all cone values
    for k in np.arange(len(conedeg)):

        #print(initial_x0_array[i])
        #print(conedeg[k])   

        #for this combination, how many spacecraft are needed to go with delta < conedeg?
        delta_current=[delta2_value3[i],delta2_value4[i], delta2_value5[i], delta2_value6[i],delta2_value7[i],delta2_value8[i],
                      delta2_value9[i], delta2_value10[i], delta2_value11[i], delta2_value12[i],
                       delta2_value13[i],delta2_value14[i],delta2_value15[i],delta2_value16[i] ]


        heatmap[i,k]=np.nan

        #print(str(np.round(delta_current,2)))
        lowestind=np.where(delta_current< conedeg[k])[0]
        #print(lowestind)

        #check if result exists
        if len(lowestind) > 0: 
            #assume that 4 is the minimum number of s/c
            if lowestind[0] > 0:
                heatmap[i,k]=nr_sc[lowestind[0]]

        #print()

    #print()


heatmap

###########################

#plot heatmap


sns.set_style('whitegrid')
sns.set_context('talk')    

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(1, 1, 1)

sns.heatmap(heatmap, annot=True, cmap='Blues', cbar=True,
            xticklabels=conedeg, yticklabels=labeling_clean,cbar_kws={'label': 'Number of spacecraft'})

ax1.set_ylabel(r'DRO $r_{min}$ [au]')
ax1.set_xlabel(r'Cone of acceptance $\Lambda$ [°] HEE longitude')

ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

plt.tight_layout()

plt.savefig(f'results/fig6_gap_analysis_heatmap.png', dpi=300,bbox_inches='tight')
plt.savefig(f'results/fig6_gap_analysis_heatmap.pdf', dpi=300,bbox_inches='tight')




# ## Figure 7 sketch for upstream monitoring
# made with affinity designer, see folder sketches/

# ## Figure ICMECAT event distribution and DROs (not used in paper)
# read ICMECAT, plot with DROs
# 

# In[40]:


url='icmecat/HELIO4CAST_ICMECAT_v23.csv'
ic=pd.read_csv(url)
print(ic.keys())

#get indices for each target
imes=np.where(ic.sc_insitu=='MESSENGER')[0]
ivex=np.where(ic.sc_insitu=='VEX')[0]
iwin=np.where(ic.sc_insitu=='Wind')[0]
imav=np.where(ic.sc_insitu=='MAVEN')[0]
ijun=np.where(ic.sc_insitu=='Juno')[0]

ista=np.where(ic.sc_insitu=='STEREO-A')[0]
istb=np.where(ic.sc_insitu=='STEREO-B')[0]
ipsp=np.where(ic.sc_insitu=='PSP')[0]
isol=np.where(ic.sc_insitu=='SolarOrbiter')[0]
ibep=np.where(ic.sc_insitu=='BepiColombo')[0]
iuly=np.where(ic.sc_insitu=='ULYSSES')[0]


# In[41]:


sns.set_style('darkgrid')
sns.set_context('talk')    

fig, ax = plt.subplots(1,figsize=(10, 8),subplot_kw={'projection': 'polar'},dpi=100)    

fsize=15
symsize_planet=10

ax.text(0,0,'Sun', color='black', ha='center',fontsize=fsize-5,verticalalignment='top')
#ax.text(0,1.2,'Earth', color='mediumseagreen', ha='center',fontsize=fsize-5,verticalalignment='center')

# Sun
ax.scatter(0,0,s=100,c='yellow',alpha=1, edgecolors='black', linewidth=0.3)

ax.scatter(earth.lon, earth.r, s=symsize_planet, c='mediumseagreen', alpha=1,lw=0,zorder=3,marker=None, label='Earth')  
#ax.plot(venus.lon, venus.r, c='gold', alpha=1,lw=1,zorder=1, marker=None, label='Venus')  
#ax.plot(mercury.lon, mercury.r, c='grey', alpha=0.5,lw=1,zorder=3, marker=None, label='Mercury')  

ax.plot(orbits_polar[1,:,0], orbits_polar[0,:,0], dro_colors[0], linewidth=1.5, alpha=1.0, label='DRO 1 0.74 au')
ax.plot(orbits_polar[1,:,3], orbits_polar[0,:,3], dro_colors[3], linewidth=1.5, alpha=1.0, label='DRO 4 0.80 au')
ax.plot(orbits_polar[1,:,6], orbits_polar[0,:,6], dro_colors[6], linewidth=1.5, alpha=1.0, label='DRO 7 0.86 au SHIELD')
ax.plot(orbits_polar[1,:,9], orbits_polar[0,:,9], dro_colors[9], linewidth=1.5, alpha=1.0, label='DRO 10 0.92 au HENON')


####### ICMECAT events

ms=3
al=0.6

#ax.plot(np.radians(ic.mo_sc_long_heeq[iuly]),ic.mo_sc_heliodistance[iuly],'o',markersize=ms,c='brown', alpha=al, label='Ulysses')
#ax.plot(np.radians(ic.mo_sc_long_heeq[imav]),ic.mo_sc_heliodistance[imav],'o',markersize=ms,c='orangered', alpha=al, label='MAVEN')

#only inner heliosphere
ax.plot(np.radians(ic.mo_sc_long_heeq[imes]),ic.mo_sc_heliodistance[imes],'o',markersize=ms,c='coral', alpha=al,label='MESSENGER')
ax.plot(np.radians(ic.mo_sc_long_heeq[ivex]),ic.mo_sc_heliodistance[ivex],'o',markersize=ms,c='orange', alpha=al,label='Venus Express')
ax.plot(np.radians(ic.mo_sc_long_heeq[istb]),ic.mo_sc_heliodistance[istb],'o',markersize=ms,c='royalblue', alpha=al,label='STEREO-B')
ax.plot(np.radians(ic.mo_sc_long_heeq[ijun]),ic.mo_sc_heliodistance[ijun],'o',markersize=ms,c='black',markerfacecolor='yellow',alpha=al,label='Juno')

#ax3.plot(ic.mo_sc_heliodistance[ijun],ic.mo_bmean[ijun],'o', c='black',markerfacecolor='yellow', alpha=al,ms=ms, label='Juno')

ax.plot(np.radians(ic.mo_sc_long_heeq[ista]),ic.mo_sc_heliodistance[ista],'o',markersize=ms, c='red', alpha=al, label='STEREO-A')
ax.plot(np.radians(ic.mo_sc_long_heeq[iwin]),ic.mo_sc_heliodistance[iwin],'o',markersize=ms, c='mediumseagreen', alpha=al, label='Wind')
ax.plot(np.radians(ic.mo_sc_long_heeq[ipsp]),ic.mo_sc_heliodistance[ipsp],'o',markersize=ms, c='black', alpha=al,label='Parker Solar Probe')
ax.plot(np.radians(ic.mo_sc_long_heeq[isol]),ic.mo_sc_heliodistance[isol],'o',markersize=ms, c='black',markerfacecolor='white', alpha=al, label='Solar Orbiter')
ax.plot(np.radians(ic.mo_sc_long_heeq[ibep]),ic.mo_sc_heliodistance[ibep],'o',markersize=ms, c='darkblue',markerfacecolor='lightgrey', alpha=al, label='BepiColombo')

plt.legend(loc=2,fontsize=10)
#1 au circle
ax.plot(np.deg2rad(np.arange(0,360)),np.zeros(360)+1,lw=1,alpha=0.8,linestyle='--',c='black', marker=None)
ax.plot(np.zeros(11),np.arange(0,1.1,0.1),c='k',lw=1,alpha=0.8,linestyle='--')


degrees = np.arange(-60,60,5)
ax.set_xticks(np.radians(degrees))
ax.set_xticklabels([f'{d}°' for d in degrees], fontsize=15)

ax.set_rgrids(np.arange(0.2,1.5,0.1),('0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.1','1.2','1.3'),angle=50, fontsize=10)


ax.set_theta_zero_location('E')
ax.set_thetamin(60)      # Start angle in degrees
ax.set_thetamax(-60)
ax.set_ylim(0, 1.3) 


ax.legend(bbox_to_anchor=(0.9, 0.9), loc='upper left',fontsize=8)
#plt.figtext(0.8,0.1,f' {nr_sc} DRO spacecraft', color='black', ha='left',fontsize=fsize-4, style='italic')
plt.figtext(0.05,0.01,'Austrian Space Weather Office   GeoSphere Austria', color='black', ha='left',fontsize=fsize-4, style='italic')
plt.figtext(0.99,0.01,'helioforecast.space', color='black', ha='right',fontsize=fsize-4, style='italic')
plt.tight_layout()

plt.savefig(f'results/dro_all_icme_polar_zoom.png', dpi=300,bbox_inches='tight')
plt.savefig(f'results/dro_all_icme_polar_zoom.pdf', dpi=300,bbox_inches='tight')


# ## Figure 8 ICMECAT event distribution and radial longitude domain
# 

# In[42]:


sns.set_style('whitegrid')
sns.set_context('talk')    

fig, ax = plt.subplots(1,figsize=(15, 12),subplot_kw={'projection': 'polar'},dpi=100)    

fsize=15
symsize_planet=10

ax.scatter(earth.lon, earth.r, s=symsize_planet, c='mediumseagreen', alpha=1,lw=0,zorder=3,marker=None, label='Earth')  
#1 au circle
ax.plot(np.deg2rad(np.arange(0,360)),np.zeros(360)+1,lw=1,alpha=0.8,linestyle='--',c='black', marker=None)
ax.plot(np.zeros(11),np.arange(0,1.1,0.1),c='k',lw=1,alpha=0.8,linestyle='--')

ax.plot(orbits_polar[1,:,0], orbits_polar[0,:,0], dro_colors[0], linewidth=2.5, alpha=1.0, label='DRO 1 0.74 au')
ax.plot(orbits_polar[1,:,3], orbits_polar[0,:,3], dro_colors[3], linewidth=2.5, alpha=1.0, label='DRO 4 0.80 au')
ax.plot(orbits_polar[1,:,6], orbits_polar[0,:,6], dro_colors[6], linewidth=2.5, alpha=1.0, label='DRO 7 0.86 au SHIELD')
ax.plot(orbits_polar[1,:,9], orbits_polar[0,:,9], dro_colors[9], linewidth=2.5, alpha=1.0, label='DRO 10 0.92 au HENON')

plot_icmecat=True

if plot_icmecat:

    ####### ICMECAT events

    ms=10
    al=0.6

    #ax.plot(np.radians(ic.mo_sc_long_heeq[iuly]),ic.mo_sc_heliodistance[iuly],'o',markersize=ms,c='brown', alpha=al, label='Ulysses')
    #ax.plot(np.radians(ic.mo_sc_long_heeq[imav]),ic.mo_sc_heliodistance[imav],'o',markersize=ms,c='orangered', alpha=al, label='MAVEN')

    #only inner heliosphere
    ax.plot(np.radians(ic.mo_sc_long_heeq[imes]),ic.mo_sc_heliodistance[imes],'o',markersize=ms,c='k',markerfacecolor='coral', alpha=al,label='MESSENGER')
    ax.plot(np.radians(ic.mo_sc_long_heeq[ivex]),ic.mo_sc_heliodistance[ivex],'o',markersize=ms,c='black', markerfacecolor='orange', alpha=al,label='Venus Express')
    ax.plot(np.radians(ic.mo_sc_long_heeq[istb]),ic.mo_sc_heliodistance[istb],'o',markersize=ms,c='royalblue', alpha=al,label='STEREO-B')
    ax.plot(np.radians(ic.mo_sc_long_heeq[ijun]),ic.mo_sc_heliodistance[ijun],'o',markersize=ms,c='black',markerfacecolor='yellow',alpha=al,label='Juno')

    #ax3.plot(ic.mo_sc_heliodistance[ijun],ic.mo_bmean[ijun],'o', c='black',markerfacecolor='yellow', alpha=al,ms=ms, label='Juno')

    ax.plot(np.radians(ic.mo_sc_long_heeq[ista]),ic.mo_sc_heliodistance[ista],'o',markersize=ms, c='white',markerfacecolor='tomato', alpha=al, label='STEREO-A')
    ax.plot(np.radians(ic.mo_sc_long_heeq[iwin]),ic.mo_sc_heliodistance[iwin],'o',markersize=ms, c='mediumseagreen', alpha=al, label='Wind')
    ax.plot(np.radians(ic.mo_sc_long_heeq[ipsp]),ic.mo_sc_heliodistance[ipsp],'o',markersize=ms, c='black', alpha=al,label='Parker Solar Probe')
    ax.plot(np.radians(ic.mo_sc_long_heeq[isol]),ic.mo_sc_heliodistance[isol],'o',markersize=ms, c='black',markerfacecolor='white', alpha=1.0, label='Solar Orbiter')
    ax.plot(np.radians(ic.mo_sc_long_heeq[ibep]),ic.mo_sc_heliodistance[ibep],'o',markersize=ms, c='darkblue',markerfacecolor='grey', alpha=0.8, label='BepiColombo')

    #plt.legend(loc=2,fontsize=10)
    #1 au circle
    #ax.plot(np.deg2rad(np.arange(0,360)),np.zeros(360)+1,lw=1,alpha=0.8,linestyle='--',c='black', marker=None)
    #ax.plot(np.zeros(11),np.arange(0,1.1,0.1),c='k',lw=1,alpha=0.8,linestyle='--')

####################### CALCULATE radial longitudinal difference map
#radial longitude difference

def diff_radial_longitudinal(R, THETA):

    # get as input r in au, theta as longitude in radians
    rf=1-R
    lf=(2*np.pi*R)/360*np.abs(np.rad2deg(THETA))
    return rf-lf

r_min = 0.3
r_max = 1.0
n_r = 500
n_t = 500

#make a polar grid
r     = np.linspace(r_min, r_max, n_r)
theta = np.linspace(-np.pi/5, np.pi/5, n_t) # in radians, up to 40°
R, THETA = np.meshgrid(r, theta)

F=diff_radial_longitudinal(R,THETA)

CMAP = "plasma" #inferno, magma, viridis, plasma
#CMAP = 'Greys_r'

level1=np.arange(-1.1,1.2,0.1)
cf = ax.contourf(THETA, R, F, levels=level1, cmap=CMAP, alpha=0.8,zorder=0, vmin=-1, vmax=1)
#cl = ax.contour(THETA, R, F, levels=15, colors="white", linewidths=0.35, alpha=0.35)
#zero level
ax.contour(THETA, R, F, levels=[0], colors="black", linewidths=1.0, alpha=0.9)

cbar = plt.colorbar(cf, ax=ax,location='right',shrink=0.5,pad=0, aspect=15)
cbar.set_label('RLD [au]',fontsize=10)
cbar.ax.tick_params(labelsize=12)
ticks = cbar.get_ticks()

degrees = np.arange(-35,40,5)
ax.set_xticks(np.radians(degrees))
ax.set_xticklabels([f'{d}°' for d in degrees], fontsize=15)

#ax.set_rgrids(np.arange(0.4,1.5,0.1),('0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.1','1.2','1.3'),angle=50, fontsize=10, zorder=5)
#rtick_locs=np.arange(0.4,1.2,0.1)
#rtick_labels=('0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0')
#ax.set_rgrids(rtick_locs,rtick_labels)#,angle=50, fontsize=5, zorder=5)

#ax.set_yticklabels([])
#for loc, lbl in zip(rtick_locs, rtick_labels):
#    ax.text(0., loc, lbl, ha='center', va='top', fontsize=16)

ax.set_theta_zero_location('E')
ax.set_thetamin(35)      # Start angle in degrees
ax.set_thetamax(-35)
##plt.title('Planet and simulated DRO positions 2028 Jan 1 - 2030 Jan 1')
#ax.set_ylim(0, 1.3) 

# cutout in r
ax.set_rmin(0.30)
ax.set_rmax(1.10)
ax.set_rorigin(0)

ax.legend(bbox_to_anchor=(0.01, 0.99), loc='upper left',fontsize=12)

plt.tight_layout()   
plt.savefig(f'results/fig8_RLD.png', dpi=200,bbox_inches='tight')
plt.savefig(f'results/fig8_RLD.pdf', dpi=200,bbox_inches='tight')


# - For each event position, get RLD value (for fun)
# - Radial longitudinal diff (RLD) plot map - every ICME event can be assigned an RLD number (make RLD statistics); check how many are in different domains, definitely a gap east of Earth for the SHIELD HENON orbits

# In[18]:


##for future work, select all events first in domain -35 to +35° and between 0 to 1 au
#all solar orbiter events
#slon=np.deg2rad(ic.mo_sc_long_heeq[isol])
#sr=ic.mo_sc_heliodistance[isol]
#slon

#s_rld=diff_radial_longitudinal(sr,slon)
#print(s_rld)
#plt.plot(s_rld,'o')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




