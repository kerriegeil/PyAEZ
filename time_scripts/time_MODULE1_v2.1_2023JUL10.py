# Author: K. Geil
# Date: 09/2023
# Description: get compute time for all functions in pyaez module 1 as it exists in the repo on 10 JULY 2023

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
gdal.UseExceptions()

#############################################################################
# SET UP
# update the appropriate settings below (everything up to "Main Code") 
# for every execution and user 
#############################################################################

##################
# Kerrie laptop
##################
work_dir = r'C://Users/kerrie/Documents/01_LocalCode/repos/PyAEZ/' # path to your PyAEZ repo
v_folder = r'pyaez2.1/pyaez2.1_2023JUL10/'     # path to the correct version directory
out_path = work_dir+'time_scripts/results/' # path for saving output data
# china
data_dir = r'C://Users/kerrie/Documents/02_LocalData/pyAEZ_input_data/china/npy/' # path to your climate data
maskfile = r'C://Users/kerrie/Documents/02_LocalData/pyAEZ_input_data/china/tif/mask.tif'
elevfile = r'C://Users/kerrie/Documents/02_LocalData/pyAEZ_input_data/china/tif/elev.tif'
soilfile = r'C://Users/kerrie/Documents/02_LocalData/pyAEZ_input_data/china/tif/soil_terrain_lulc_china_08333.tif'

##################
# Kerrie desktop
##################
# work_dir = r'K:/projects/unfao/pyaez_gaez/repos/PyAEZ_kerrie/PyAEZ/' 
# v_folder = r'pyaez2.1/pyaez2.1_2023JUL10/'
# out_path = work_dir+'time_scripts/results/'
# # china
# data_dir = r'C://Users/kerrie.WIN/Documents/data/pyAEZ_data_inputs_china_03272023/npy/' 
# maskfile = r'C://Users/kerrie.WIN/Documents/data/pyAEZ_data_inputs_china_03272023/tif/mask.tif'
# elevfile = r'C://Users/kerrie.WIN/Documents/data/pyAEZ_data_inputs_china_03272023/tif/elev.nc'
# soilfile = r'C://Users/kerrie.WIN/Documents/data/pyAEZ_data_inputs_china_03272023/tif/soil_terrain_lulc_china_08333.tif'

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# enter personal and system info for output filename
domain='china'
person='KLG' # your initials
location='home' #'HPC'#'home'#'SSC' # nickname for your location
computer='Windows10'#'Orion' #Windows10 # operating system
processor='IntelCorei7-12800H'#'400p48h'  #'IntelCorei7-12800H' #'IntelZeonW2225' # 'bigmem' # processor
ram='32GB' # RAM
test_tag='v2.1_2023JUL10' # short description of what is being timed
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Some other parameters, these shouldn't really have to change
mask_value = 0  # pixel value in admin_mask to exclude from the analysis
daily = True # Type of climate data = True: daily, False: monthly

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
        'getLGP, getLGPClassified, getLGPEquivalent',
        'getTemperatureProfile',
        'getMultiCroppingZones',
        'AirFrostIndexandPermafrostEvaluation',
        'TZoneFallowRequirement',
        'AEZClassification']   
#############################################################################
#############################################################################


#####################################
#####################################
#####################################
# MAIN CODE
#####################################
#####################################
#####################################

# Create output dir if it does not exist
isExist = os.path.exists(out_path)
if not isExist:
   os.makedirs(out_path)

sys.path.append(work_dir+v_folder) # add pyaez model to system path

# limit grid precision
lats=rio.open_rasterio(maskfile)['y'].data
lat_min = np.trunc(lats.min()*100000)/100000 # use only 5 decimal places
lat_max = np.trunc(lats.max()*100000)/100000 # use only 5 decimal places
# lat_min = 18.04167
# lat_max = 53.625
mask_path=maskfile

# name the output file
outfile=out_path+'time_results_'+domain+'_'\
    +test_tag+'_'+person+'_'+location+'_'+computer+'_'+processor+'_'+ram+'.txt'

labels=timelabels
if daily:
    labels.insert(4,'setDailyClimateData')
else:
    labels.insert(4,'setMonthlyClimateData')

# labels=labels[:-1] # take this out after soil_terrrain_lulc is prepared
results=odict.fromkeys(labels)

import ClimateRegime_v2_1 as ClimateRegime
clim_reg = ClimateRegime.ClimateRegime()
import UtilitiesCalc_v2_1 as UtilitiesCalc
obj_utilities=UtilitiesCalc.UtilitiesCalc()

starttime=timeit()  # overall timer

# Task load data
taskstart=timeit()
print('loading data')
max_temp = np.load(data_dir+'Tmax-2m365/0.npy').astype('float32')  # maximum temperature
min_temp = np.load(data_dir+'Tmin-2m365/0.npy').astype('float32')  # minimum temperature
precipitation = np.load(data_dir+'Precip365/0.npy').astype('float32')  # precipitation
rel_humidity = np.load(data_dir+'Rhum365/0.npy').astype('float32')  # relative humidity
wind_speed = np.load(data_dir+'Wind-2m365/0.npy').astype('float32') # wind speed measured at two meters
short_rad = np.load(data_dir+'Srad365/0.npy').astype('float32')  # shortwave radiation
mask=gdal.Open(maskfile).ReadAsArray()
elevation=gdal.Open(elevfile).ReadAsArray()
soil_terrain_lulc=gdal.Open(soilfile).ReadAsArray()

results['input data load to mem time']=timeit()-taskstart

# Task Module 1 class object set up
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

# Task module 1 functions

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
# Length of Growing Periods
#####################################
if (timetests==['all']) or ('getLGP, getLGPClassified, getLGPEquivalent' in timetests):
    print('getLGP, getLGPClassified, getLGPEquivalent')
    taskstart=timeit()
    # call functions
    lgp = clim_reg.getLGP( Sa = 100, D=1. )
    lgp_class = clim_reg.getLGPClassified(lgp)
    lgp_equv = clim_reg.getLGPEquivalent()
    results['getLGP, getLGPClassified, getLGPEquivalent']=timeit()-taskstart

#####################################
# Temperature Profile
#####################################
if (timetests==['all']) or ('getTemperatureProfile' in timetests):
    print('getTemperatureProfile')
    taskstart=timeit()
    tprofile = clim_reg.getTemperatureProfile() # call function
    results['getTemperatureProfile']=timeit()-taskstart    

#####################################
# Multi-Cropping Zone
#####################################
if (timetests==['all']) or ('getMultiCroppingZones' in timetests):
    print('getMultiCroppingZones')
    taskstart=timeit()
    # call function
    multi_crop = clim_reg.getMultiCroppingZones(tclimate, lgp, lgpt5, lgpt10, tsum0, tsum10)
    results['getMultiCroppingZones']=timeit()-taskstart

#####################################
# Air Frost Index and Permafrost Evaluation
#####################################
if (timetests==['all']) or ('AirFrostIndexandPermafrostEvaluation' in timetests):
    print('AirFrostIndexandPermafrostEvaluation')
    taskstart=timeit()
    # call function
    permafrost_eval = clim_reg.AirFrostIndexandPermafrostEvaluation()
    results['AirFrostIndexandPermafrostEvaluation']=timeit()-taskstart

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
if (timetests==['all']) or ('AEZClassification' in timetests):
    print('AEZClassification')
    taskstart=timeit()
    aez = clim_reg.AEZClassification(tclimate, lgp, lgp_equv, lgpt5, soil_terrain_lulc, permafrost_eval[1]) # call function
    results['AEZClassification']=timeit()-taskstart

# overall timer
if timetests==['all']:
    results['module1 all funcs']=timeit()-starttime

# write file
print('writing results to',outfile)
with open(outfile,'w') as f:
    for key,val in results.items():
        if val != None:
            f.write('%60s %15s %4s\n' % (str(key),str(round(val,3)),'s'))
        else:
            f.write('%60s %15s %4s\n' % (str(key),str(val),'s'))
