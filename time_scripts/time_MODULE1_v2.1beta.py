# Author: K. Geil, from NB1_ClimateRegime.ipynb
# Date: 03/2023
# Description: time the different functions of pyaez module 1 (Climate Regime)

# import packages
# import matplotlib.pyplot as plt
import os
import sys
import numpy as np
try:
    from osgeo import gdal
except:
    import gdal
import rioxarray as rio
import xarray as xr
from time import time as timeit
from collections import OrderedDict as odict

#############################################################################
# SET UP
# these are the things that may need to be updated per user
#############################################################################

# set up directories
# HPC Orion
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# change this to your PyAEZ directory
# work_dir = '/work/hpc/users/kerrie/UN_FAO/repos/PyAEZ/'
# #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# # these are the same for everyone
# data_dir = '/work/hpc/datasets/un_fao/pyaez/china_8110/daily/npy/' # subset for no antarctica, 1800 lats
# maskfile = '/work/hpc/datasets/un_fao/pyaez/china_static/netcdf/mask.nc'# subset for no antarctica, 1800 lats
# elevfile = '/work/hpc/datasets/un_fao/pyaez/china_static/tif/elev.tif'
# out_path = work_dir+'time_scripts/results/' # path for saving output data

# Kerrie laptop
work_dir = 'C://Users/kerrie/Documents/01_LocalCode/repos/PyAEZ/' # path to your PyAEZ repo
data_dir = 'C://Users/kerrie/Documents/02_LocalData/pyAEZ_input_data/china/' # path to your data
maskfile = 'C://Users/kerrie/Documents/02_LocalData/pyAEZ_input_data/china/mask.nc'# subset for no antarctica, 1800 lats
elevfile = 'C://Users/kerrie/Documents/02_LocalData/pyAEZ_input_data/china/elev.nc'
out_path = work_dir+'time_scripts/results/' # path for saving output data

# Create output dir if it does not exist
isExist = os.path.exists(out_path)
if not isExist:
   os.makedirs(out_path)

sys.path.append(work_dir+'pyaez_v2.1beta/') # add pyaez model to system path
import ClimateRegime as ClimateRegime
clim_reg = ClimateRegime.ClimateRegime()
import UtilitiesCalc as UtilitiesCalc
obj_utilities=UtilitiesCalc.UtilitiesCalc()

# Define the Area-Of-Interest's geographical extents
lats=rio.open_rasterio(maskfile)['y'].data
lat_min = np.trunc(lats.min()*100000)/100000 # use only 5 decimal places
lat_max = np.trunc(lats.max()*100000)/100000 # use only 5 decimal places
mask_path=maskfile
mask_value = 0  # pixel value in admin_mask to exclude from the analysis
daily = True # Type of climate data = True: daily, False: monthly

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# enter personal and system info for output filename
domain='china'
person='KLG' # your initials
location='home' #'SSC' # nickname for your location
computer='Windows10' #Windows10 # operating system
processor='IntelCorei7-12800H' #'IntelZeonW2225' # 'bigmem' # processor
ram='32GB' # RAM
test_tag='v2_1beta' # short description of what is being timed

outfile=out_path+'time_results_'+domain+'_'\
    +test_tag+'_'+person+'_'+location+'_'+computer+'_'+processor+'_'+ram+'.txt'
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# choose which functions should be timed
# if you want a subset of timelabels, then specify them here as timetests
# otherwise, use ['all']
timetests=['all']
# timetests=['module1 all funcs',
#         'input data load to mem time',
#         'setStudyAreaMask',
#         'setLocationTerrainData',
#         'getThermalClimate',
#         'getThermalZone',
#         'getThermalLGP0, getThermalLGP5, getThermalLGP10',
#         'getTemperatureSum0, getTemperatureSum5, getTemperatureSum10',
#         'getTemperatureProfile',        
#         'getLGP, getLGPClassified, getLGPEquivalent',
#         'getMultiCroppingZones',
#         'AirFrostIndexandPermafrostEvaluation',
#         'TZoneFallowRequirement']

# labels for timing each function
# don't change these
timelabels=['module1 all funcs',
        'input data load to mem time',
        'setStudyAreaMask',
        'setLocationTerrainData',
        'getThermalClimate',
        'getThermalZone',
        'getThermalLGP0, getThermalLGP5, getThermalLGP10',
        'getTemperatureSum0, getTemperatureSum5, getTemperatureSum10',
        'getTemperatureProfile',
        'getLGP, getLGPClassified, getLGPEquivalent',
        'getMultiCroppingZones',
        'AirFrostIndexandPermafrostEvaluation',
        'TZoneFallowRequirement',
        'AEZClassification']

labels=timelabels
if daily:
    labels.insert(4,'setDailyClimateData')
else:
    labels.insert(4,'setMonthlyClimateData')

labels=labels[:-1] # take this out after soil_terrrain_lulc is prepared
results=odict.fromkeys(labels)
#############################################################################
#############################################################################


#####################################
#####################################
#####################################
# MAIN CODE
#####################################
#####################################
#####################################

starttime=timeit()

# Open the data files, this is quick
taskstart=timeit()
print('loading data')
# max_temp = np.load(data_dir+'Tmax-2m365/0.npy').astype('float32')  # maximum temperature
# min_temp = np.load(data_dir+'Tmin-2m365/0.npy').astype('float32')  # minimum temperature
# precipitation = np.load(data_dir+'Precip365/0.npy').astype('float32')  # precipitation
# rel_humidity = np.load(data_dir+'Rhum365/0.npy').astype('float32')  # relative humidity
# wind_speed = np.load(data_dir+'Wind-2m365/0.npy').astype('float32') # wind speed measured at two meters
# short_rad = np.load(data_dir+'Srad365/0.npy').astype('float32')  # shortwave radiation
# mask=gdal.Open(maskfile).ReadAsArray()
# elevation=gdal.Open(elevfile).ReadAsArray()

# subset for quick testing that the script doesn't error 
# max_temp=max_temp[600:700,1000:1100,:]
# min_temp=min_temp[600:700,1000:1100,:]
# precipitation=precipitation[600:700,1000:1100,:]
# rel_humidity=rel_humidity[600:700,1000:1100,:]
# wind_speed=wind_speed[600:700,1000:1100,:]
# short_rad=short_rad[600:700,1000:1100,:]
# mask=mask[600:700,1000:1100]
# elevation=elevation[600:700,1000:1100]

max_temp = xr.open_dataset(data_dir+'tmax_daily_8110.nc')['tmax'].transpose('lat','lon','doy').data
min_temp = xr.open_dataset(data_dir+'tmin_daily_8110.nc')['tmin'].transpose('lat','lon','doy').data
precipitation = xr.open_dataset(data_dir+'prcp_daily_8110.nc')['prcp'].transpose('lat','lon','doy').data
rel_humidity = xr.open_dataset(data_dir+'relh_daily_8110.nc')['relh'].transpose('lat','lon','doy').data
wind_speed = xr.open_dataset(data_dir+'wspd_daily_8110.nc')['wspd'].transpose('lat','lon','doy').data
short_rad = xr.open_dataset(data_dir+'srad_daily_8110.nc')['srad'].transpose('lat','lon','doy').data
mask=xr.open_dataset(data_dir+'mask.nc')['mask'].data
elevation=xr.open_dataset(data_dir+'elev.nc')['elev'].data

results['input data load to mem time']=timeit()-taskstart

# Module 1 class object set up
taskstart=timeit()
print('setting mask')
clim_reg.setStudyAreaMask(mask, mask_value)
results['setStudyAreaMask']=timeit()-taskstart

taskstart=timeit()
print('setting grid')
clim_reg.setLocationTerrainData(lat_min, lat_max, elevation)
# clim_reg.setLocationTerrainData(lats, elevation)
results['setLocationTerrainData']=timeit()-taskstart

print('creating climate data class object')
if daily:
    taskstart=timeit()
    clim_reg.setDailyClimateData(
        min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity)
    results['setDailyClimateData']=timeit()-taskstart
else:
    taskstart=timeit()
    clim_reg.setMonthlyClimateData(
        min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity)
    results['setMonthlyClimateData']=timeit()-taskstart

del(min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity) # free mem


#####################################
# Thermal Climate
#####################################
if (timetests==['all']) or ('getThermalClimate' in timetests):
    print('getThermalClimate')
    taskstart=timeit()
    tclimate = clim_reg.getThermalClimate() # call function
    results['getThermalClimate']=timeit()-taskstart

#####################################
# Thermal Zone
#####################################
if (timetests==['all']) or ('getThermalZone' in timetests):
    print('getThermalZone')
    taskstart=timeit()
    tzone = clim_reg.getThermalZone() # call function
    results['getThermalZone']=timeit()-taskstart

#####################################
# Thermal Length of Growing Period (LGP)
#####################################
if (timetests==['all']) or ('getThermalLGP0, getThermalLGP5, getThermalLGP10' in timetests):
    print('getThermalLGP0, getThermalLGP5, getThermalLGP10')
    taskstart=timeit()
    # call functions
    lgpt0 = clim_reg.getThermalLGP0()
    lgpt5 = clim_reg.getThermalLGP5()
    lgpt10 = clim_reg.getThermalLGP10()
    results['getThermalLGP0, getThermalLGP5, getThermalLGP10']=timeit()-taskstart

#####################################
# Temperature Sum
#####################################
if (timetests==['all']) or ('getTemperatureSum0, getTemperatureSum5, getTemperatureSum10' in timetests):
    print('getTemperatureSum0, getTemperatureSum5, getTemperatureSum10')
    taskstart=timeit()
    # call functions
    tsum0 = clim_reg.getTemperatureSum0()
    tsum5 = clim_reg.getTemperatureSum5()
    tsum10 = clim_reg.getTemperatureSum10()
    results['getTemperatureSum0, getTemperatureSum5, getTemperatureSum10']=timeit()-taskstart

#####################################
# Temperature Profile
#####################################
if (timetests==['all']) or ('getTemperatureProfile' in timetests):
    print('getTemperatureProfile')
    taskstart=timeit()
    tprofile = clim_reg.getTemperatureProfile() # call function
    results['getTemperatureProfile']=timeit()-taskstart

#####################################
# Length of Growing Periods
#####################################
if (timetests==['all']) or ('getLGP, getLGPClassified, getLGPEquivalent' in timetests):
    print('getLGP, getLGPClassified, getLGPEquivalent')
    taskstart=timeit()
    # call functions
    lgp = clim_reg.getLGP( Sa = 100 )
    lgp_class = clim_reg.getLGPClassified(lgp)
    lgp_equv = clim_reg.getLGPEquivalent()
    results['getLGP, getLGPClassified, getLGPEquivalent']=timeit()-taskstart

#####################################
# Multi-Cropping Zone
#####################################
if (timetests==['all']) or ('getMultiCroppingZones' in timetests):
    print('getMultiCroppingZones')
    taskstart=timeit()
    # call function
    multi_crop = clim_reg.getMultiCroppingZones(tclimate, lgp, lgpt5, lgpt10, tsum0, tsum10)
    results['getMultiCroppingZones']=timeit()-taskstart

    multi_crop_rainfed = multi_crop[0]  # for rainfed conditions
    multi_crop_irr = multi_crop[1]  # for irrigated conditions

#####################################
# Air Frost Index and Permafrost Evaluation
#####################################
if (timetests==['all']) or ('AirFrostIndexandPermafrostEvaluation' in timetests):
    print('AirFrostIndexandPermafrostEvaluation')
    taskstart=timeit()
    # call function
    permafrost_eval = clim_reg.AirFrostIndexandPermafrostEvaluation()
    results['AirFrostIndexandPermafrostEvaluation']=timeit()-taskstart

    frost_index = permafrost_eval[0]
    permafrost = permafrost_eval[1]

#####################################
# Fallow Period Requirement
#####################################
if (timetests==['all']) or ('TZoneFallowRequirement' in timetests):
    print('TZoneFallowRequirement')
    taskstart=timeit()
    tzone_fallow = clim_reg.TZoneFallowRequirement(tzone) # call function
    results['TZoneFallowRequirement']=timeit()-taskstart

# #####################################
# # Agro-ecological zone classification
# #####################################
# if (timetests==['all']) or ('AEZClassification' in timetests):
#     print('AEZClassification')
#     taskstart=timeit()
#     aez = clim_reg.AEZClassification(
#         tclimate, lgp, lgp_equv, lgpt5, soil_terrain_lulc, permafrost) # call function
#     results['AEZClassification']=timeit()-taskstart

    # # save png plot
    # fig = plt.figure()
    # plt.imshow(aez, cmap=plt.get_cmap('rainbow', 59), vmin=0, vmax=59)
    # plt.title('Agro-ecological Zonation')
    # plt.colorbar()
    # plt.savefig(out_path+"aez.png",
    #             bbox_inches="tight", dpi=300)

    # # save tif data
    # obj_utilities.saveRaster(
    #     mask_path, out_path+'aez.tif', aez)

if timetests==['all']:
    results['module1 all funcs']=timeit()-starttime

print('writing results to',outfile)
with open(outfile,'w') as f:
    for key,val in results.items():
        if val != None:
            f.write('%60s %15s %4s\n' % (str(key),str(round(val,3)),'s'))
        else:
            f.write('%60s %15s %4s\n' % (str(key),str(val),'s'))
