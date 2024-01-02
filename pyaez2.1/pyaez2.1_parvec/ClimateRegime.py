"""
PyAEZ version 2.1.0 (June 2023)
This ClimateRegime Class read/load and calculates the agro-climatic indicators
required to run PyAEZ.  
2021: N. Lakmal Deshapriya
2022/2023: Swun Wunna Htet and Kittiphon Boonma

"""

import numpy as np
# from pyaez import UtilitiesCalc, ETOCalc, LGPCalc
import UtilitiesCalc as UtilitiesCalc  
import ETOCalc as ETOCalc # KLG
import LGPCalc as LGPCalc # KLG
from collections import OrderedDict
# from time import time as timer
import sys


# Initiate ClimateRegime Class instance
class ClimateRegime(object):
    def setParallel(self,var3D,parallel=False,nchunks=None,reduce_mem_used=False):

        if parallel:
            # we parallelize by chunking longitudes
            self.parallel=True
            self.chunk2D,self.chunk3D,self.chunksize3D_MB,self.nchunks=UtilitiesCalc.UtilitiesCalc().setChunks(nchunks,var3D.shape,reduce_mem_used)    
        else:
            self.parallel=False
            self.chunk3D=None
            self.chunk2D=None
            self.chunksize3D_MB=None
            self.nchunks=None

    # def setLocationTerrainData(self, lats, elevation):  
        # option to take all lats as an input  
        # why not take in an array of all latitudes here instead of regenerating lats from min/max
        # could avoid slight shifts/precision problems with the grid         
    # def setLocationTerrainData(self, lat_min, lat_max, location, elevation): 
    def setLocationTerrainData(self, lat_min, lat_max, location, elevation,lats,lons): 
        """Load geographical extents and elevation data in to the Class, 
           and create a latitude map

        Args:
            lat_min (float): the minimum latitude of the AOI in decimal degrees
            lat_max (float): the maximum latitude of the AOI in decimal degrees
            elevation (2D NumPy): elevation map in metres
        """
        if self.parallel:
            import dask
            self.elevation = elevation.rechunk(chunks=self.chunk2D)
        else:        
            self.elevation = elevation

        self.im_height = elevation.shape[0]
        self.im_width = elevation.shape[1]
        self.latitude = UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D).generateLatitudeMap(lat_min, lat_max, location, self.im_height, self.im_width)   
        # self.latitude = UtilitiesCalc.UtilitiesCalc().generateLatitudeMap(lats, location)  # option to take all lats as an input
        self.lats=lats
        self.lons=lons
        
    
    def setStudyAreaMask(self, admin_mask, no_data_value):
        """Set clipping mask of the area of interest (optional)

        Args:
            admin_mask (2D NumPy/Binary): mask to extract only region of interest
            no_data_value (int): pixels with this value will be omitted during PyAEZ calculations
        """    
        if self.parallel:
            # import dask            
            self.im_mask = admin_mask.rechunk(chunks=self.chunk2D).astype('int8')
        else:
            self.im_mask = admin_mask.astype('int8')

        self.nodata_val = no_data_value
        self.set_mask = True
  

    def setMonthlyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):
        """Load MONTHLY climate data into the Class and calculate the Reference Evapotranspiration (ETo)

        Args:
            min_temp (3D NumPy): Monthly minimum temperature [Celcius]
            max_temp (3D NumPy): Monthly maximum temperature [Celcius]
            precipitation (3D NumPy): Monthly total precipitation [mm/day]
            short_rad (3D NumPy): Monthly solar radiation [W/m2]
            wind_speed (3D NumPy): Monthly windspeed at 2m altitude [m/s]
            rel_humidity (3D NumPy): Monthly relative humidity [percentage decimal, 0-1]
        """    
        #################################################################
        ##### THIS FUNCTION HAS NOT BEEN UPDATED TO RUN IN PARALLEL #####
        #################################################################
        self.doy_start=1 
        self.doy_end=min_temp.shape[2] 
        
        rel_humidity[rel_humidity > 0.99] = 0.99
        rel_humidity[rel_humidity < 0.05] = 0.05
        short_rad[short_rad < 0] = 0
        wind_speed[wind_speed < 0] = 0

        # Interpolate monthly to daily data
        obj_utilities = UtilitiesCalc.UtilitiesCalc()

        meanT_monthly = (min_temp+max_temp)/2
                
        # Sea-level adjusted mean temperature
        self.meanT_daily_sealevel = self.meanT_daily + np.tile(np.reshape(self.elevation/100*0.55, (self.im_height,self.im_width,1)), (1,1,365))
        # P over PET ratio(to eliminate nan in the result, nan is replaced with zero)
        self.P_by_PET_daily = np.nan_to_num(self.totalPrec_daily / self.pet_daily)
        self.set_monthly = True

    def setDailyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):
        """Load DAILY climate data into the Class and calculate the Reference Evapotranspiration (ETo)

        Args:
            min_temp (3D NumPy): Daily minimum temperature [Celcius]
            max_temp (3D NumPy): Daily maximum temperature [Celcius]
            precipitation (3D NumPy): Daily total precipitation [mm/day]
            short_rad (3D NumPy): Daily solar radiation [W/m2]
            wind_speed (3D NumPy): Daily windspeed at 2m altitude [m/s]
            rel_humidity (3D NumPy): Daily relative humidity [percentage decimal, 0-1]
        """
        if self.parallel:
            import dask.array as da
            import dask

        self.doy_start=1  
        self.doy_end=min_temp.shape[2] 

        self.set_monthly = False

        if self.parallel:
            precipitation = precipitation.rechunk(chunks=self.chunk3D)

            ### CALCULATE PET_DAILY ###
            # chunk and delay inputs, compute pet_daily in parallel with dask.delayed
            min_temp=min_temp.rechunk(chunks=self.chunk3D)
            max_temp=max_temp.rechunk(chunks=self.chunk3D)
            tmn_delay=min_temp.to_delayed().ravel()
            tmx_delay=max_temp.to_delayed().ravel()

            short_rad=short_rad.rechunk(chunks=self.chunk3D) # chunk
            short_rad=da.where(short_rad < 0, 0, short_rad)  # elim negatives
            short_rad = short_rad*3600.*24./1000000.#*0.0864## convert units
            srad_delay=short_rad.to_delayed().ravel()

            wind_speed=wind_speed.rechunk(chunks=self.chunk3D)     # chunk
            wind_speed=da.where(wind_speed < 0.5, 0.5, wind_speed) # elim negative and small values
            wind_delay=wind_speed.to_delayed().ravel() 

            rel_humidity=rel_humidity.rechunk(chunks=self.chunk3D)        # chunk
            rel_humidity=da.where(rel_humidity > 0.99, 0.99,rel_humidity) # elim high values
            rel_humidity=da.where(rel_humidity < 0.05, 0.05,rel_humidity) # elim low values
            rh_delay=rel_humidity.to_delayed().ravel()

            lat_delay=self.latitude.to_delayed().ravel()
            elev_delay=self.elevation.to_delayed().ravel()  

            zipvars=zip(lat_delay,elev_delay,tmn_delay,tmx_delay,wind_delay,srad_delay,rh_delay)
            obj_eto=ETOCalc.ETOCalc() 
            task_list=[dask.delayed(obj_eto.calculateETO)(self.doy_start,self.doy_end,lat,el,tmn,tmx,u,srad,rh) for lat,el,tmn,tmx,u,srad,rh in zipvars]
            # print('in ClimateRegime, computing pet_daily in parallel')
            result_chunks=dask.compute(*task_list)
            self.pet_daily=np.concatenate(result_chunks,axis=1) 
            del result_chunks,tmn_delay,tmx_delay,srad_delay,short_rad,wind_delay,wind_speed,rh_delay,rel_humidity,lat_delay,elev_delay,zipvars,task_list
        else:
            ### CALCULATE PET_DAILY ###
            rel_humidity[rel_humidity > 0.99] = 0.99
            rel_humidity[rel_humidity < 0.05] = 0.05
            short_rad[short_rad < 0] = 0
            short_rad = short_rad*0.0864
            wind_speed[wind_speed < 0.5] = 0.5
            
            obj_eto=ETOCalc.ETOCalc()
            self.pet_daily= obj_eto.calculateETO(self.doy_start,self.doy_end,self.latitude,self.elevation,min_temp,max_temp,wind_speed,short_rad,rel_humidity)
        
        ### CALCULATE MEANT_DAILY ###
        self.meanT_daily = 0.5*(min_temp + max_temp)   
        
        obj_utilities = UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D)   

        if self.parallel:
            ### CALCULATE MEANT_MONTHLY_SEALEVEL ###
            # (we only ever use monthly mean sea level so it's pointless to carry daily data in RAM)
            meanT_daily_sealevel = self.meanT_daily + (da.broadcast_to(self.elevation[:,:,np.newaxis]/100*.55,(self.im_height,self.im_width,self.doy_end)))
            # print('in ClimateRegime, agg daily to meanT_monthly_sealevel in parallel')
            self.meanT_monthly_sealevel = obj_utilities.averageDailyToMonthly(meanT_daily_sealevel)   
            del meanT_daily_sealevel
            if self.set_mask:
                mask_monthly=da.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,12))
                self.meanT_monthly_sealevel = da.where(mask_monthly,self.meanT_monthly_sealevel,np.float32(np.nan)).compute()
            
            ### CALCULATE P_BY_PET_MONTHLY ####
            # same for P_by_PET_monthly, we only use monthly later
            # chunk and delay inputs
            # pr_delayed=precipitation.rechunk(chunks=self.chunk3D).to_delayed().ravel()
            # pet_delayed=da.from_array(self.pet_daily,chunks=self.chunk3D).to_delayed().ravel()
            # mask_delayed=da.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,12),chunks=(self.chunk2D[0],self.chunk2D[1],12)).to_delayed().ravel()
            pr=precipitation.rechunk(chunks=self.chunk3D) # dask array
            pet=da.from_array(self.pet_daily,chunks=self.chunk3D)  # dask array
            # mask3D=da.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,12),chunks=(self.chunk2D[0],self.chunk2D[1],12)) # numpy array
            # mask_monthly=mask_monthly.compute() # numpy array
            with np.errstate(divide='ignore', invalid='ignore'):
                P_by_PET_daily = np.nan_to_num(pr/pet)  #dask array
                P_by_PET_monthly = obj_utilities.averageDailyToMonthly(P_by_PET_daily)  # compute monthly values (np array)

            if self.set_mask:
                mask_monthly=mask_monthly.compute() # numpy array
                self.P_by_PET_monthly = np.where(mask_monthly,P_by_PET_monthly,np.float32(np.nan)) # implement mask
            else:
                self.P_by_PET_monthly=P_by_PET_monthly
            del pr, pet, P_by_PET_daily,P_by_PET_monthly


            # # put the comp inside a function that we can delay
            # def calc_P_by_PET_monthly(pet,pr,mask,obj_util):
            #     with np.errstate(divide='ignore', invalid='ignore'):
            #         pet=np.where(pet==0,np.float32(np.nan),pet) # replace inf in the P_by result with nan
            #         P_by_PET_daily = pr/pet # daily values
            #         # P_by_PET_daily = np.nan_to_num(pr/pet)
            #         P_by_PET_monthly = obj_util.averageDailyToMonthly(P_by_PET_daily)  # monthly values
            #         P_by_PET_monthly = np.where(mask,P_by_PET_monthly,np.float32(np.nan)) # implement mask
            #     return P_by_PET_monthly
            
            # task_list=[dask.delayed(calc_P_by_PET_monthly)(pet_chunk,pr_chunk,mask_chunk,obj_utilities) for pet_chunk,pr_chunk,mask_chunk in zip(pr_delayed,pet_delayed,mask_delayed)]
            # # print('in ClimateRegime, computing P_by_PET in parallel')
            # results=dask.compute(*task_list)
            # # concatenate chunks
            # self.P_by_PET_monthly=np.concatenate(results,axis=1) 
            # del results,task_list,pr_delayed,pet_delayed,mask_delayed

        else:
            ### CALCULATE MEANT_MONTHLY_SEALEVEL ###
            meanT_daily_sealevel = self.meanT_daily + np.expand_dims(self.elevation/100*.55,axis=2)      
            self.meanT_monthly_sealevel = obj_utilities.averageDailyToMonthly(meanT_daily_sealevel)   
            del meanT_daily_sealevel          

            ### CALCULATE P_BY_PET_MONTHLY ####
            with np.errstate(invalid='ignore',divide='ignore'):
                # pet=np.where(self.pet_daily==0,np.float32(np.nan),self.pet_daily) # replace inf in the P_by result with nan
                # P_by_PET_daily = precipitation / pet
                P_by_PET_daily = np.nan_to_num(precipitation / self.pet_daily)
                self.P_by_PET_monthly = obj_utilities.averageDailyToMonthly(P_by_PET_daily)  # monthly values

            if self.set_mask:
                mask_monthly=np.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,12))
                self.meanT_monthly_sealevel = np.where(mask_monthly,self.meanT_monthly_sealevel,np.float32(np.nan))                  
                self.P_by_PET_monthly = np.where(mask_monthly,self.P_by_PET_monthly,np.float32(np.nan))

        ### SET DAILY VARIABLES TO CLASS OBJECT ###
        self.maxT_daily = max_temp
        self.totalPrec_daily = precipitation
        del precipitation

        ### CALCULATE MONTHLY AND ANNUAL VALUES ###
        # adding these other small things to RAM will save compute time later

        # get the mask if there is one        
        if self.set_mask:
            if self.parallel:
                mask=self.im_mask.compute()
                # already computed (dask-->np) mask_monthly above
                # mask_monthly=mask_monthly.compute()
            else:
                mask=self.im_mask

        # monthly mean T and precip
        # print('in ClimateRegime, computing meanT_monthly, totalPrec_monthly')     
        self.meanT_monthly = obj_utilities.averageDailyToMonthly(self.meanT_daily)
        self.totalPrec_monthly = obj_utilities.averageDailyToMonthly(self.totalPrec_daily) 
        if self.set_mask:
            self.meanT_monthly=np.where(mask_monthly,self.meanT_monthly,np.float32(np.nan))
            self.totalPrec_monthly=np.where(mask_monthly,self.totalPrec_monthly,np.float32(np.nan))

        # annual mean T
        if self.parallel:
            # print('in ClimateRegime, computing annual_Tmean')
            self.annual_Tmean = da.mean(self.meanT_daily, axis = 2).compute()      
        else:
            self.annual_Tmean = np.mean(self.meanT_daily, axis = 2)

        if self.set_mask:
            self.annual_Tmean=np.where(mask,self.annual_Tmean,np.float32(np.nan))                         

        # annual accumulated precip
        if self.parallel:
            # print('in ClimateRegime, computing annual_accPrec')
            self.annual_accPrec = da.sum(self.totalPrec_daily, axis = 2).compute()           
        else:
            self.annual_accPrec = np.sum(self.totalPrec_daily, axis = 2) 

        if self.set_mask:
            self.annual_accPrec=np.where(mask,self.annual_accPrec,np.float32(np.nan))                                

        # annual accumulated pet
        # print('in ClimateRegime, computing annual_accPET')
        self.annual_accPET = np.sum(self.pet_daily, axis = 2)
        if self.set_mask:
            self.annual_accPET=np.where(mask,self.annual_accPET,np.float32(np.nan))                                


    def getThermalClimate(self):
        """Classification of rainfall and temperature seasonality into thermal climate classes

        Returns:
            2D NumPy: Thermal Climate classification
        """        
        # Note that currently, this thermal climate is designed only for the northern hemisphere, southern hemisphere is not implemented yet.
        if self.parallel:
            import dask

        thermal_climate = np.zeros((self.im_height,self.im_width),dtype='int8')   

        # get monthly/annual variables
        meanT_monthly_sealevel=self.meanT_monthly_sealevel
        meanT_monthly=self.meanT_monthly        
        P_by_PET_monthly=self.P_by_PET_monthly
        prsum=self.annual_accPrec

        # other things we need to assign thermal_climate values   
        # compute them here for readability below   
        summer_PET0=P_by_PET_monthly[:,:,3:9].sum(axis=2) # Apr-Sep   
        JFMSON=[0,1,2,9,10,11]   
        winter_PET0=P_by_PET_monthly[:,:,JFMSON].sum(axis=2) # Oct-Mar   
        min_sealev_meanT=meanT_monthly_sealevel.min(axis=2)   
        Ta_diff=meanT_monthly_sealevel.max(axis=2) - meanT_monthly_sealevel.min(axis=2)   
        # Ta_diff=meanT_monthly.max(axis=2) - meanT_monthly.min(axis=2)   
        meanT=meanT_monthly.mean(axis=2)   
        nmo_ge_10C=(meanT_monthly_sealevel >= 10).sum(axis=2)   
        
        if self.chunk3D:
            # print('in ClimateRegime, computing latitude in parallel')
            latitude=self.latitude.compute()
        else:
            latitude=self.latitude

        # Tropics   
        # Tropical lowland   
        thermal_climate=np.where((min_sealev_meanT>18.) & (Ta_diff<15.) & (meanT>=20.),1,thermal_climate)   
        # Tropical highland   
        thermal_climate=np.where((min_sealev_meanT>18.) & (Ta_diff<15.) & (meanT<20.) & (thermal_climate==0),2,thermal_climate)   
        
        # SubTropic   
        # Subtropics Low Rainfall   
        thermal_climate=np.where((min_sealev_meanT>5.) & (nmo_ge_10C>=8) & (prsum<250) & (thermal_climate==0),5,thermal_climate)   
        # Subtropics Summer and Winter Rainfall   
        thermal_climate=np.where((min_sealev_meanT>5.) & (nmo_ge_10C>=8) & (prsum>=250)& (latitude>=0) & (summer_PET0>=winter_PET0) & (thermal_climate==0),3,thermal_climate)   
        thermal_climate=np.where((min_sealev_meanT>5.) & (nmo_ge_10C>=8) & (prsum>=250)& (latitude>=0) & (summer_PET0<winter_PET0) & (thermal_climate==0),4,thermal_climate)   

        thermal_climate=np.where((min_sealev_meanT>5.) & (nmo_ge_10C>=8) & (prsum>=250)& (latitude<0) & (summer_PET0>=winter_PET0) & (thermal_climate==0),4,thermal_climate)   
        thermal_climate=np.where((min_sealev_meanT>5.) & (nmo_ge_10C>=8) & (prsum>=250)& (latitude<0) & (summer_PET0<winter_PET0) & (thermal_climate==0),3,thermal_climate)   
        # Temperate   
        # Oceanic Temperate   
        thermal_climate=np.where((nmo_ge_10C>=4) & (Ta_diff<=20) & (thermal_climate==0),6,thermal_climate)   
        # Sub-Continental Temperate   
        thermal_climate=np.where((nmo_ge_10C>=4) & (Ta_diff<=35) & (thermal_climate==0),7,thermal_climate)   
        # Continental Temperate   
        thermal_climate=np.where((nmo_ge_10C>=4) & (Ta_diff>35) & (thermal_climate==0),8,thermal_climate)   
        
        # Boreal   
        # Oceanic Boreal   
        thermal_climate=np.where((nmo_ge_10C>=1) & (Ta_diff<=20) & (thermal_climate==0),9,thermal_climate)   
        # Sub-Continental Boreal   
        thermal_climate=np.where((nmo_ge_10C>=1) & (Ta_diff<=35) & (thermal_climate==0),10,thermal_climate)   
        # Continental Boreal   
        thermal_climate=np.where((nmo_ge_10C>=1) & (Ta_diff>35) & (thermal_climate==0),11,thermal_climate)   
        
        # Arctic   
        thermal_climate=np.where((thermal_climate==0),12,thermal_climate)   

        if self.set_mask:
            if self.parallel:
                # print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask
            thermal_climate=np.where(mask, thermal_climate.astype('float32'), np.float32(np.nan))   
            return thermal_climate
        else:
            return thermal_climate#.astype('float32')

    

    def getThermalZone(self):
        """The thermal zone is classified based on actual temperature which reflects 
        on the temperature regimes of major thermal climates

        Returns:
            2D NumPy: Thermal Zones classification
        """        
        if self.parallel:
            import dask
        thermal_zone = np.zeros((self.im_height,self.im_width),dtype='int8')   

        # get monthly variables
        meanT_monthly=self.meanT_monthly
        meanT_monthly_sealevel=self.meanT_monthly_sealevel

        # things we need to determine the classes
        # compute them here for readability below
        min_sealev_meanT=meanT_monthly_sealevel.min(axis=2)
        range_meanT=meanT_monthly.max(axis=2) - meanT_monthly.min(axis=2)
        meanT=meanT_monthly.mean(axis=2)
        # do we need both of the next two?
        nmo_gt_10C_sealev=(meanT_monthly_sealevel > 10).sum(axis=2)
        nmo_ge_10C_sealev=(meanT_monthly_sealevel >= 10).sum(axis=2)
        nmo_lt_5C=(meanT_monthly < 5).sum(axis=2)
        nmo_gt_10C=(meanT_monthly > 10).sum(axis=2)
        nmo_lt_10C=(meanT_monthly < 10).sum(axis=2)    

        # Tropics, warm
        thermal_zone=np.where((min_sealev_meanT>=18) & (range_meanT<15) & (meanT>20),1,thermal_zone)
        # Tropics, cool/cold/very cold
        thermal_zone=np.where((min_sealev_meanT>=18) & (range_meanT<15) & (meanT<=20) & (thermal_zone==0),2,thermal_zone)
        # Subtropics, cool
        thermal_zone=np.where((min_sealev_meanT>5) & (nmo_gt_10C_sealev>=8) & (nmo_lt_5C>=1) & (nmo_gt_10C>=4) & (thermal_zone==0),4,thermal_zone)
        # Subtropics, cold
        thermal_zone=np.where((min_sealev_meanT>5) & (nmo_gt_10C_sealev>=8) & (nmo_lt_5C>=1) & (nmo_gt_10C>=1) & (thermal_zone==0),5,thermal_zone)
        #Subtropics, very cold
        thermal_zone=np.where((min_sealev_meanT>5) & (nmo_gt_10C_sealev>=8) & (nmo_lt_10C==12) & (thermal_zone==0),6,thermal_zone)
        # Subtropics, warm/mod. cool
        thermal_zone=np.where((min_sealev_meanT>5) & (nmo_gt_10C_sealev>=8) & (thermal_zone==0),3,thermal_zone)        
        # Temperate, cool
        thermal_zone=np.where((nmo_ge_10C_sealev>=4) & (nmo_lt_5C>=1) & (nmo_gt_10C>=4) & (thermal_zone==0),7,thermal_zone)
        # Temperate, cold
        thermal_zone=np.where((nmo_ge_10C_sealev>=4) & (nmo_lt_5C>=1) & (nmo_gt_10C>=1) & (thermal_zone==0),8,thermal_zone)
        # Temperate, very cold
        thermal_zone=np.where((nmo_ge_10C_sealev>=4) & (nmo_lt_10C==12) & (thermal_zone==0),9,thermal_zone)
        # Boreal, cold
        thermal_zone=np.where((nmo_ge_10C_sealev>=1) & (nmo_lt_5C>=1) & (nmo_gt_10C>=1) & (thermal_zone==0),10,thermal_zone)
        # Boreal, very cold
        thermal_zone=np.where((nmo_ge_10C_sealev>=1) & (nmo_lt_10C==12) & (thermal_zone==0),11,thermal_zone)
        # Arctic
        thermal_zone=np.where((thermal_zone==0),12,thermal_zone)
    
        if self.set_mask:
            if self.parallel:
                # print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask            
            return np.where(mask, thermal_zone.astype('float32'), np.float32(np.nan))   
        else:
            return thermal_zone#.astype('float32')   


    def getThermalLGP0(self):
        """Calculate Thermal Length of Growing Period (LGPt) with 
        temperature threshold of 0 degree Celcius

        Returns:
            2D numpy: The accumulated number of days with daily mean 
                      temperature is above 0 degree Celcius
        """        
        if self.parallel:
            import dask

        lgpt0 = np.sum(self.meanT_daily>=0, axis=2,dtype='float32')
        if self.set_mask:
            lgpt0 = np.where(self.im_mask,lgpt0,np.float32(np.nan))
        
        if self.parallel:
            # print('in ClimateRegime, computing lgpt0 in parallel')
            lgpt0=lgpt0.compute()

        self.lgpt0=lgpt0

        return lgpt0


    def getThermalLGP5(self):
        """Calculate Thermal Length of Growing Period (LGPt) with 
        temperature threshold of 5 degree Celcius

        Returns:
            2D numpy: The accumulated number of days with daily mean 
                      temperature is above 5 degree Celcius
        """  
        if self.parallel:
            import dask

        lgpt5 = np.sum(self.meanT_daily>=5, axis=2,dtype='float32')
        if self.set_mask:
            lgpt5 = np.where(self.im_mask,lgpt5,np.float32(np.nan))
        
        if self.parallel:
            # print('in ClimateRegime, computing lgpt5 in parallel')
            lgpt5=lgpt5.compute()

        self.lgpt5 = lgpt5
        return lgpt5


    def getThermalLGP10(self):
        """Calculate Thermal Length of Growing Period (LGPt) with
        temperature threshold of 10 degree Celcius

        Returns:
            2D numpy: The accumulated number of days with daily mean
                      temperature is above 10 degree Celcius
        """
        if self.parallel:
            import dask

        lgpt10 = np.sum(self.meanT_daily >= 10, axis=2,dtype='float32')
        if self.set_mask:
            lgpt10 = np.where(self.im_mask, lgpt10, np.float32(np.nan))
        
        if self.parallel:
            # print('in ClimateRegime, computing lgpt10 in parallel')
            lgpt10=lgpt10.compute()

        self.lgpt10 = lgpt10
        return lgpt10


    def getTemperatureSum0(self):
        """Calculate temperature summation at temperature threshold 
        of 0 degree Celcius

        Returns:
            2D numpy: Accumulative daily average temperature (Ta) for days
                      when Ta is above the thresholds of 0 degree Celcius
        """
        if self.parallel:
            import dask

        tempT=np.where(self.meanT_daily<0,0,self.meanT_daily)
        tsum0 = np.round(np.sum(tempT, axis=2,dtype='float32'), decimals = 0) 

        if self.set_mask:
            tsum0 = np.where(self.im_mask, tsum0, np.float32(np.nan))
        
        if self.parallel:
            # print('in ClimateRegime, computing tsum0 in parallel')
            tsum0=tsum0.compute()
        return tsum0


    def getTemperatureSum5(self):
        """Calculate temperature summation at temperature threshold 
        of 5 degree Celcius

        Returns:
            2D numpy: Accumulative daily average temperature (Ta) for days
                      when Ta is above the thresholds of 5 degree Celcius
        """
        if self.parallel:
            import dask

        tempT=np.where(self.meanT_daily<5,0,self.meanT_daily)
        tsum5 = np.round(np.sum(tempT, axis=2,dtype='float32'), decimals = 0) 

        if self.set_mask: 
            tsum5 = np.where(self.im_mask, tsum5, np.float32(np.nan))

        if self.parallel:
            # print('in ClimateRegime, computing tsum5 in parallel')
            tsum5=tsum5.compute()
        return tsum5
        

    def getTemperatureSum10(self):
        """Calculate temperature summation at temperature threshold 
        of 10 degree Celcius

        Returns:
            2D numpy: Accumulative daily average temperature (Ta) for days
                      when Ta is above the thresholds of 10 degree Celcius
        """
        if self.parallel:
            import dask

        tempT=np.where(self.meanT_daily<10,0,self.meanT_daily)
        tsum10 = np.round(np.sum(tempT, axis=2,dtype='float32'), decimals = 0) 

        if self.set_mask: 
            tsum10 = np.where(self.im_mask, tsum10, np.float32(np.nan))

        if self.parallel:
            # print('in ClimateRegime, computing tsum10 in parallel')
            tsum10=tsum10.compute()            
        return tsum10


    def getTemperatureProfile(self):
        """Classification of temperature ranges for temperature profile

        Returns:
            2D NumPy: 18 2D arrays [A1-A9, B1-B9] correspond to each Temperature Profile class [days]
        """    
        if self.parallel:
            import dask
            import dask.array as da

        # a nested ordered dictionary containing info needed to compute for each t profile class
        tclass_info = OrderedDict({ 'A1':{'tendency':'warming','lim_lo':30,'lim_hi':999},
                                    'A2':{'tendency':'warming','lim_lo':25,'lim_hi':30},
                                    'A3':{'tendency':'warming','lim_lo':20,'lim_hi':25},
                                    'A4':{'tendency':'warming','lim_lo':15,'lim_hi':20},
                                    'A5':{'tendency':'warming','lim_lo':10,'lim_hi':15},
                                    'A6':{'tendency':'warming','lim_lo':5,'lim_hi':10},
                                    'A7':{'tendency':'warming','lim_lo':0,'lim_hi':5},
                                    'A8':{'tendency':'warming','lim_lo':-5,'lim_hi':0},
                                    'A9':{'tendency':'warming','lim_lo':-999,'lim_hi':-5},
                                    'B1':{'tendency':'cooling','lim_lo':30,'lim_hi':999},
                                    'B2':{'tendency':'cooling','lim_lo':25,'lim_hi':30},
                                    'B3':{'tendency':'cooling','lim_lo':20,'lim_hi':25},
                                    'B4':{'tendency':'cooling','lim_lo':15,'lim_hi':20},
                                    'B5':{'tendency':'cooling','lim_lo':10,'lim_hi':15},
                                    'B6':{'tendency':'cooling','lim_lo':5,'lim_hi':10},
                                    'B7':{'tendency':'cooling','lim_lo':0,'lim_hi':5},
                                    'B8':{'tendency':'cooling','lim_lo':-5,'lim_hi':0},
                                    'B9':{'tendency':'cooling','lim_lo':-999,'lim_hi':-5} })

        # put the computations inside a function
        def sum_ndays_per_tprof_class(diff,meanT,tendency,lim_lo,lim_hi,mask):
            if tendency=='warming':
                tclass_ndays = np.sum( (diff>0)&(meanT>=lim_lo)&(meanT<lim_hi), axis=2 ) 
            if tendency=='cooling':
                tclass_ndays = np.sum( (diff<0)&(meanT>=lim_lo)&(meanT<lim_hi), axis=2 )

            # apply mask
            tclass_ndays=np.where(mask, tclass_ndays.astype('float32'), np.float32(np.nan))
            return tclass_ndays 

        # create a mask of all 1's if the user doesn't provide a mask
        if self.set_mask:
            mask=self.im_mask 
        else:
            mask=np.ones((self.im_height,self.im_width),dtype='int')

        # compute 5th order spline smoothed daily temperature and attach to class object   
        obj_utilities = UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D)   
        try:
            self.interp_daily_temp=obj_utilities.smoothDailyTemp(self.doy_start,self.doy_end, mask, self.meanT_daily)
        except:
            sys.exit('Not enough RAM available to complete calculation. Try restarting notebook with parallel=True')

        if self.parallel:
            meanT_first=da.from_array(self.interp_daily_temp,chunks=self.chunk3D)
            meanT_diff=da.diff(meanT_first,n=1,axis=2,append=meanT_first[:,:,0:1]).rechunk(self.chunk3D)    

            # delay the input data so it's copied once instead of at each call of the function
            meanT_diff=meanT_diff.to_delayed().ravel()
            meanT_first=meanT_first.to_delayed().ravel()
            mask=mask.to_delayed().ravel() 

            # in a regular (non-delayed) loop, call delayed function and compile list of compute tasks
            task_list=[] 
            for class_inputs in tclass_info.values():
                for d,t,m in zip(meanT_diff,meanT_first,mask):
                    # task=sum_ndays_per_tprof_class(d,t,class_inputs['tendency'],class_inputs['lim_lo'],class_inputs['lim_hi'],m)
                    task=dask.delayed(sum_ndays_per_tprof_class)(d,t,class_inputs['tendency'],class_inputs['lim_lo'],class_inputs['lim_hi'],m)
                    task_list.append(task)

            # compute tasks in parallel
            # this returns a list of arrays in the same order as tclass_info
            # print('in ClimateRegime, computing temperature profiles in parallel')
            data_out=dask.compute(*task_list)   

            # concatenate chunks
            tprofiles=[]
            for i,key in enumerate(tclass_info.keys()):
                # print(key)
                tprofiles.append(np.concatenate(data_out[i*self.nchunks:i*self.nchunks+self.nchunks],axis=1))
        else:
            meanT_diff=np.diff(self.interp_daily_temp,n=1,axis=2,append=self.interp_daily_temp[:,:,0:1])

            tprofiles=[]
            for class_inputs in tclass_info.values():
                tprofiles.append(sum_ndays_per_tprof_class(meanT_diff,self.interp_daily_temp,class_inputs['tendency'],class_inputs['lim_lo'],class_inputs['lim_hi'],mask))
        
        return tprofiles   


    def getLGP(self, Sa=100., D=1.):
        """Calculate length of growing period (LGP)

        Args:
            Sa (float, optional): Available soil moisture holding capacity [mm/m]. Defaults to 100..
            D (float, optional): Rooting depth. Defaults to 1..

        Returns:
           2D NumPy: Length of Growing Period
        """   
        if self.parallel:
            import dask
            import dask.array as da

        # constants
        Txsnm = 0. 
        Fsnm = 5.5
        kc_list = np.array([0.0, 0.1, 0.2, 0.5, 1.0],dtype='float32')  

        if self.parallel:
            # generalized workflow:
            # 1) prep inputs chunked like (all y, x chunk, all days)
            # 2) call daily loop function LGPCalc.EtaCalc on chunks, return lgp_tot in chunks (in RAM)
            # 3) concat lgp_tot to shape (ny,nx)

            # set up larger chunks for quicker processing
            # start=timer()
            nlons=int(np.ceil(self.chunk2D[1]*4))  # consider adding a user override for this
            bigchunk2D=(-1,nlons)
            bigchunk3D=(-1,nlons,-1)
            nchunks=int(np.ceil(self.im_width/nlons))    
            print('using larger chunks:',nchunks,'total chunks instead of',self.nchunks)      

            # build task graph for istart0,istart1,p
            lgpt5=da.from_array(self.lgpt5,chunks=bigchunk2D)
            istart0,istart1=LGPCalc.rainPeak(self.meanT_daily.rechunk(chunks=bigchunk3D),lgpt5)
            # self.istart0=istart0
            # self.istart1=istart1
            ng=da.zeros(self.pet_daily.shape,chunks=bigchunk3D,dtype='float32')
            pet=da.from_array(self.pet_daily,chunks=bigchunk3D)

            # compute eta_class
            # the task graph for eta_class is so complex that it's faster 
            # to compute it outside of any loops and hold in RAM
            lgpt5_3D=da.broadcast_to(self.lgpt5[:,:,np.newaxis].astype('float16'),(self.im_height,self.im_width,self.doy_end)).rechunk(chunks=bigchunk3D).to_delayed().ravel()
            mask_3D=da.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,self.doy_end)).rechunk(chunks=bigchunk3D).to_delayed().ravel()
            Tmean=self.meanT_daily.rechunk(chunks=bigchunk3D).astype('float16').to_delayed().ravel()
            Tmax=self.maxT_daily.rechunk(chunks=bigchunk3D).astype('float16').to_delayed().ravel()
            zipvars=zip(mask_3D,lgpt5_3D,Tmean,Tmax)
            task_list=[dask.delayed(LGPCalc.Eta_class)(m,l5,Tbar,Tmx,Txsnm) for m,l5,Tbar,Tmx in zipvars]
            results=dask.compute(*task_list)
            eta_class=np.concatenate(results,axis=1)
            del lgpt5_3D,mask_3D,Tmean,Tmax,zipvars,task_list,results
            # self.eta_class=eta_class[:,:,0]
            # build task graph for islgp
            islgp=da.where(self.meanT_daily>=5,np.int8(1),np.int8(0)).rechunk(chunks=bigchunk3D)   

            # chunk all inputs to big chunks as defined above
            # these are all lazy dask arrays
            lgpt5_c=lgpt5  # already chunked
            mask_c=self.im_mask.rechunk(chunks=bigchunk2D)
            istart0_c = istart0  # already chunked
            istart1_c = istart1  # already chunked
            Sb_old_c = da.zeros((self.im_height,self.im_width),chunks=bigchunk2D,dtype='float32')
            Wb_old_c = da.zeros((self.im_height,self.im_width),chunks=bigchunk2D,dtype='float32')
            Pet365_c = pet # already chunked
            p_c = LGPCalc.psh(ng,pet)
            eta_class_c=da.from_array(eta_class,chunks=bigchunk3D)
            Tx365_c = self.maxT_daily.rechunk(chunks=bigchunk3D)
            Pcp365_c = self.totalPrec_daily.rechunk(chunks=bigchunk3D)
            islgp_c = islgp  # already chunked
            # task_time=timer()-start
            # print('time spent on prepping vars',task_time) 

            # start=timer()
            # this is not a normal way to compute with dask
            # our functions are so complicated that allowing dask to automate the parallel 
            # computation is much slower and/or crashes due to high memory use
            # here we loop thru chunks one at a time, compute the inputs ahead 
            # of time to reduce passing many/large task graphs, and call the EtaCalc 
            # func (which includes some parallelism) on each chunk, then concat the resulting lgp_tot chunks
            results=[]
            for i in range(nchunks):
                if i%10 == 0: print('loop',(i+1),'of',nchunks,', this message prints every 10 chunks')
                # convert input chunks to numpy arrays in memory
                mask_np=mask_c.blocks[0,i].compute().copy()
                Tx365_np=Tx365_c.blocks[0,i,0].compute().copy()
                islgp_np=islgp_c.blocks[0,i,0].compute().copy()
                Pcp365_np=Pcp365_c.blocks[0,i,0].compute().copy()
                Pet365_np=Pet365_c.blocks[0,i,0].compute().copy()
                Wb_old_np=Wb_old_c.blocks[0,i].compute().copy()
                Sb_old_np=Sb_old_c.blocks[0,i].compute().copy()
                istart0_np=istart0_c.blocks[0,i].compute().copy()
                istart1_np=istart1_c.blocks[0,i].compute().copy()
                lgpt5_np=lgpt5_c.blocks[0,i].compute().copy()
                eta_class_np=eta_class_c.blocks[0,i].compute().copy()
                p_np=p_c.blocks[0,i,0].compute().copy()

                # compute lgp_tot in chunks
                results.append(LGPCalc.EtaCalc(mask_np,Tx365_np,islgp_np,Pcp365_np,\
                                            Txsnm,Fsnm,Pet365_np,Wb_old_np,Sb_old_np,\
                                            istart0_np,istart1_np,Sa,D,p_np,kc_list,\
                                            lgpt5_np,eta_class_np,self.doy_start,self.doy_end,self.parallel))
            # task_time=timer()-start
            # print('time spent in compute',task_time)  

            # del self.pet_daily # free up RAM

            # concatenate result chunks
            # start=timer()
            lgp_tot=np.concatenate(results,axis=1)

            # ####################################################################
            # ####### dont forget to delete this #################################
            # ####### DEBUGGING LGPCalc.py, exporting different variables ########
            # ####################################################################
            # import pandas as pd
            # import xarray as xr
            # out_dir=r'C://Users/kerrie/Documents/01_LocalCode/repos/PyAEZ/pyaez2.1/pyaez2.1_parvec/debug/'
            # varname='ETA'
            # filename=varname+'_parvec21_LGPCalc.nc'
            # year=1980            
            # time=pd.date_range(str(year)+'-01-01',str(year)+'-12-31',freq='D')
            # if len(time) != lgp_tot.shape[-1]:
            #     time=time[~(time==str(year)+'-02-29')]

            # # metadata for output data files
            # timeattrs={'standard_name':'time','long_name':'time','axis':'T'}
            # latattrs={'standard_name':'latitude','long_name':'latitude','units':'degrees_north','axis':'Y'}
            # lonattrs={'standard_name':'longitude','long_name':'longitude','units':'degrees_east','axis':'X'}

            # # encoding info for writing netcdf files
            # time_encoding={'calendar':'standard','units':'days since 1900-01-01 00:00:00','_FillValue':None}
            # lat_encoding={'_FillValue':None}
            # lon_encoding={'_FillValue':None}
            # var_encoding = {'zlib':True,'dtype':'float32'}

            # var_xr=xr.DataArray(lgp_tot,
            #         dims=['lat','lon','time'],
            #         coords={'lat':('lat',y),'lon':('lon',x),'time':('time',time)}).astype('float32')
            # var_xr.name=varname
            # var_xr['lat'].attrs=latattrs
            # var_xr['lon'].attrs=lonattrs
            # var_xr.attrs={'description':'ETA from LGPCalc.py'}
            # var_xr=var_xr.to_dataset()
            # # write mask netcdf file
            # var_xr.to_netcdf(out_dir+filename,
            #             encoding={'lat':lat_encoding,'lon':lon_encoding,varname:var_encoding})  
            # ####################################################################
            # ####################################################################
            # ####################################################################
            # ####################################################################



            # task_time=timer()-start
            # print('time spent on lgp_tot concat',task_time)   
            # return lgp_tot.astype('float32')  
            # return eta_class
            if self.set_mask:
                return np.where(self.im_mask.compute(), lgp_tot.astype('float32'), np.float32(np.nan))   
            else:
                return lgp_tot.astype('float32')
            # return lgp_tot.astype('float32')   
            # self.ETA_check=lgp_tot
        else:
            try:
                istart0,istart1=LGPCalc.rainPeak(self.meanT_daily,self.lgpt5)
                ng=np.zeros(self.pet_daily.shape,dtype='float32')
                p = LGPCalc.psh(ng,self.pet_daily)
                self.p=p

                # compute eta_class
                lgpt5_3D=np.broadcast_to(self.lgpt5[:,:,np.newaxis].astype('float16'),(self.im_height,self.im_width,self.doy_end))
                mask_3D=np.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,self.doy_end))
                Tmean=self.meanT_daily.astype('float16')
                Tmax=self.maxT_daily.astype('float16')
                eta_class = LGPCalc.Eta_class(mask_3D,lgpt5_3D,Tmean,Tmax,Txsnm)
                del lgpt5_3D,mask_3D,Tmean,Tmax

                islgp=np.where(self.meanT_daily>=5,np.int8(1),np.int8(0))

                Sb_old = np.zeros((self.im_height,self.im_width),dtype='float32')
                Wb_old = np.zeros((self.im_height,self.im_width),dtype='float32')

                # compute lgp_tot                     
                lgp_tot=LGPCalc.EtaCalc(self.im_mask,self.maxT_daily,islgp,self.totalPrec_daily,\
                                        Txsnm,Fsnm,self.pet_daily,Wb_old,Sb_old,istart0,istart1,\
                                        Sa,D,p,kc_list,self.lgpt5,eta_class,self.doy_start,self.doy_end,self.parallel)

                if self.set_mask:
                    return np.where(self.im_mask, lgp_tot.astype('float32'), np.float32(np.nan))   
                else:
                    return lgp_tot.astype('float32')                                           
            except:
                sys.exit('Not enough RAM available to complete calculation. Try restarting notebook with parallel=True')


    def getLGPClassified(self, lgp): # Original PyAEZ source code
        """This function calculates the classification of moisture regime using LGP.

        Args:
            lgp (2D NumPy): Length of Growing Period

        Returns:
            2D NumPy: Classified Length of Growing Period
        """        
        if self.parallel:
            import dask

        lgp_class = np.zeros(lgp.shape,dtype='float32')   

        lgp_class=np.where(lgp>=365,7,lgp_class) # Per-humid
        lgp_class=np.where((lgp>=270)&(lgp<365),6,lgp_class) # Humid
        lgp_class=np.where((lgp>=180)&(lgp<270),5,lgp_class) # Sub-humid
        lgp_class=np.where((lgp>=120)&(lgp<180),4,lgp_class) # Moist semi-arid
        lgp_class=np.where((lgp>=60)&(lgp<120),3,lgp_class) # Dry semi-arid
        lgp_class=np.where((lgp>0)&(lgp<60),2,lgp_class) # Arid
        lgp_class=np.where(lgp<=0,1,lgp_class) # Hyper-arid

        if self.set_mask:
            if self.parallel:
                # print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask
            return np.where(mask, lgp_class.astype('float32'), np.float32(np.nan))
        else:
            return lgp_class
        
        
    def getLGPEquivalent(self): 
        """Calculate the Equivalent LGP 

        Returns:
            2D NumPy: LGP Equivalent 
        """        
        if self.parallel:
            import dask

        moisture_index = self.annual_accPrec/self.annual_accPET

        lgp_equv = 14.0 + 293.66*moisture_index - 61.25*moisture_index*moisture_index
        lgp_equv=np.where(moisture_index>2.4,366,lgp_equv)

        if self.set_mask:
            if self.parallel:
                # print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask
            return np.where(mask, lgp_equv.astype('float32'), np.float32(np.nan))   
        else:
            return lgp_equv

        # '''
        # Existing Issue: The moisture index calculation is technical aligned with FORTRAN routine, 
        # results are still different from GAEZ; causing large discrepancy. 
        # Overall, there are no changes with the calculation steps and logics.
        # '''


    def TZoneFallowRequirement(self, tzone):
        """
        The function calculates the temperature for fallow requirements which 
        requires thermal zone to classify. If mask is on, the function will
        mask out pixels by the mask layer. (NEW FUNCTION)

        Args:
        tzone : a 2-D numpy array
            THERMAL ZONE.

        Returns:
        A 2-D numpy array, corresponding to thermal zone for fallow requirement.

        """
        if self.parallel:
            import dask
            
        tzonefallow = np.zeros((self.im_height, self.im_width), dtype= 'int8')        
        annual_Tmean=self.annual_Tmean
        max_meanTmonthly=self.meanT_monthly.max(axis=2)

        # tropical thermal zones
        # Class 1: tropics, mean annual T > 25 deg C
        tzonefallow=np.where(((tzone==1)|(tzone==2))&(annual_Tmean>25),1,tzonefallow)   
        # Class 2: tropics, mean annual T 20-25 deg C   
        tzonefallow=np.where(((tzone==1)|(tzone==2))&(annual_Tmean>20)&(tzonefallow==0),2,tzonefallow)   
        # Class 3: tropics, mean annual T 15-20 deg C   
        tzonefallow=np.where(((tzone==1)|(tzone==2))&(annual_Tmean>15)&(tzonefallow==0),3,tzonefallow)   
        # Class 4: tropics, mean annual T < 15 deg C   
        tzonefallow=np.where(((tzone==1)|(tzone==2))&(annual_Tmean<=15)&(tzonefallow==0),4,tzonefallow)   
        # non-tropical zones   
        # Class 5: mean T of the warmest month > 20 deg C   
        tzonefallow=np.where((tzone!=1)&(tzone!=2)&(max_meanTmonthly>20)&(tzonefallow==0),5,tzonefallow)   
        tzonefallow=np.where((tzone!=1)&(tzone!=2)&(max_meanTmonthly<=20)&(tzonefallow==0),6,tzonefallow)     
                            
        if self.set_mask:
            if self.parallel:
                # print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask            
            return np.where(mask, tzonefallow.astype('float32'), np.float32(np.nan))  
        else:
            return tzonefallow#.astype('float32')      
  
   
    def AirFrostIndexandPermafrostEvaluation(self):
        """
        The function calculates the air frost index which is used for evaluation of 
        occurrence of continuous or discontinuous permafrost condtions executed in 
        GAEZ v4. Two outputs of numerical air frost index and classified reference
        permafrost zones are returned. If mask layer is inserted, the function will
        automatically mask user-defined pixels out of the calculation 

        Returns:
        air_frost_index/permafrost : a python list: [air frost number, permafrost classes]

        """
        if self.parallel:
            import dask
                    
        permafrost = np.zeros((self.im_height, self.im_width), dtype='int8') 

        if self.parallel:
            meanT_gt_0 = self.meanT_daily
            meanT_le_0 = meanT_gt_0.copy()            
        else:
            meanT_gt_0 = self.meanT_daily.copy()
            meanT_le_0 = self.meanT_daily.copy()
        
        meanT_gt_0=np.where(meanT_gt_0 <=0, 0, meanT_gt_0) # removing all negative temperatures for summation
        meanT_le_0=np.where(meanT_le_0 >0, 0, meanT_le_0) # removing all positive temperatures for summation  
        ddt = np.sum(meanT_gt_0, axis = 2,dtype='float32') # thawing index
        ddf = - np.sum(meanT_le_0, axis = 2,dtype='float32')  # freezing index
        fi = np.sqrt(ddf)/(np.sqrt(ddf) + np.sqrt(ddt)) # frost index

        # classify the permafrost zones (Reference: GAEZ v4 model documentation: Pg35 -37)
        permafrost=np.where(fi>0.625,1,permafrost) # Continuous Permafrost Class   
        permafrost=np.where((fi>0.57)&(fi<=0.625),2,permafrost) # Discontinuous Permafost Class   
        permafrost=np.where((fi>0.495)&(fi<=0.57),3,permafrost) # Sporadic Permafrost Class   
        permafrost=np.where(fi<=0.495,4,permafrost) # No Permafrost Class  

        fi = np.nan_to_num(fi)

        if self.set_mask:
            if self.parallel:
                # print('in ClimateRegime, computing mask in parallel')                
                fi=np.where(self.im_mask, fi, np.float32(np.nan)).compute()
                permafrost=np.where(self.im_mask, permafrost.astype('float32'), np.float32(np.nan)).compute()
            else:
                fi=np.where(self.im_mask, fi, np.float32(np.nan))
                permafrost=np.where(self.im_mask, permafrost.astype('float32'), np.float32(np.nan))
            return [fi,permafrost]
        else:
            return [fi.astype('float32'), permafrost.astype('float32')]   

    
    def AEZClassification(self, tclimate, lgp, lgp_equv, lgpt_5, soil_terrain_lulc, permafrost):
        """The AEZ inventory combines spatial layers of thermal and moisture regimes 
        with broad categories of soil/terrain qualities.

        Args:
            tclimate (2D NumPy): Thermal Climate classes
            lgp (2D NumPy): Length of Growing Period
            lgp_equv (2D NumPy): LGP Equivalent
            lgpt_5 (2D NumPy): Thermal LGP of Ta>5˚C
            soil_terrain_lulc (2D NumPy): soil/terrain/special land cover classes (8 classes)
            permafrost (2D NumPy): Permafrost classes

        Returns:
           2D NumPy: 57 classes of AEZ
        """        
        #1st step: reclassifying the existing 12 classes of thermal climate into 6 major thermal climate.
        # Class 1: Tropics, lowland
        # Class 2: Tropics, highland
        # Class 3: Subtropics
        # Class 4: Temperate Climate
        # Class 5: Boreal Climate
        # Class 6: Arctic Climate
        aez_tclimate = np.zeros((self.im_height, self.im_width), dtype='int8')
        # tropics highland
        aez_tclimate=np.where((tclimate==1),1,aez_tclimate)
        aez_tclimate=np.where((tclimate==2),2,aez_tclimate)
        aez_tclimate=np.where((tclimate==3),3,aez_tclimate)
        aez_tclimate=np.where((tclimate==4),3,aez_tclimate)
        aez_tclimate=np.where((tclimate==5),3,aez_tclimate)
        # grouping all the temperate classes into a single class 4    
        aez_tclimate=np.where((tclimate==6),4,aez_tclimate)
        aez_tclimate=np.where((tclimate==7),4,aez_tclimate)
        aez_tclimate=np.where((tclimate==8),4,aez_tclimate)
        # grouping all the boreal classes into a single class 5
        aez_tclimate=np.where((tclimate==9),5,aez_tclimate)
        aez_tclimate=np.where((tclimate==10),5,aez_tclimate)
        aez_tclimate=np.where((tclimate==11),5,aez_tclimate)
        # changing the arctic class into class 6
        aez_tclimate=np.where((tclimate==12),6,aez_tclimate)


        # 2nd Step: Classification of Thermal Zones
        aez_tzone = np.zeros((self.im_height, self.im_width), dtype='int8')
        # things we need for the conditional statements
        nmo_ge_10=np.sum(self.meanT_monthly>=10,axis=2)
        nmo_lt_10=np.sum(self.meanT_monthly<10,axis=2)
        nmo_ge_5=np.sum(self.meanT_monthly>=5,axis=2)
        temp_acc_10deg = np.where(self.meanT_daily<10,0,self.meanT_daily).sum(axis=2)
        nday_gt_20=np.sum(self.meanT_daily>20,axis=2)
        if self.parallel:
            temp_acc_10deg=temp_acc_10deg.compute()
            nday_gt_20=nday_gt_20.compute()
        # Warm Tzone (TZ1)
        aez_tzone=np.where((nmo_ge_10==12)&(self.annual_Tmean>=20),1,aez_tzone)
        # Moderately cool Tzone (TZ2)
        aez_tzone=np.where((nmo_ge_5==12)&(nmo_ge_10>=8)&(aez_tzone==0),2,aez_tzone)
        # TZ3 Moderate
        aez_tzone=np.where((aez_tclimate==4)&(nmo_ge_10>=5)&(nday_gt_20>=75)&(temp_acc_10deg>3000)&(aez_tzone==0),3,aez_tzone)
        # TZ4 Cool
        aez_tzone=np.where((nmo_ge_10>=4)&(self.annual_Tmean>=0)&(aez_tzone==0),4,aez_tzone)
        # TZ5 Cold
        aez_tzone=np.where((nmo_ge_10>=1)&(nmo_ge_10<=3)&(self.annual_Tmean>=0)&(aez_tzone==0),5,aez_tzone)
        # TZ6 Very cold
        aez_tzone=np.where((nmo_lt_10==12)|(self.annual_Tmean<0)&(aez_tzone==0),6,aez_tzone)


        # 3rd Step: Creation of Temperature Regime Classes
        # Temperature Regime Class Definition
        # 1 = Tropics, lowland (TRC1)
        # 2 = Tropics, highland (TRC2)
        # 3 = Subtropics, warm (TRC3)
        # 4 = Subtropics, moderately cool (TRC4)
        # 5 = Subtropics, cool (TRC5)
        # 6 = Temperate, moderate (TRC6)
        # 7 = Temperate, cool (TRC7)
        # 8 = Boreal, cold, no continuous or discontinuous occurrence of permafrost (TRC8)
        # 9 = Boreal, cold, with continuous or discontinuous occurrence of permafrost (TRC9)
        # 10 = Arctic, very cold (TRC10)
        aez_temp_regime = np.zeros((self.im_height, self.im_width), dtype='int8')
        aez_temp_regime = np.where((aez_tclimate==1)&(aez_tzone==1),1,aez_temp_regime) # Tropics, lowland
        aez_temp_regime = np.where((aez_tclimate==2)&((aez_tzone==2)|(aez_tzone==4))&(aez_temp_regime==0),2,aez_temp_regime) # Tropics, highland
        aez_temp_regime = np.where((aez_tclimate==3)&(aez_tzone==1)&(aez_temp_regime==0),3,aez_temp_regime) # Subtropics, warm
        aez_temp_regime = np.where((aez_tclimate==3)&(aez_tzone==2)&(aez_temp_regime==0),4,aez_temp_regime) # Subtropics,moderate cool
        aez_temp_regime = np.where((aez_tclimate==3)&(aez_tzone==4)&(aez_temp_regime==0),5,aez_temp_regime) # Subtropics,cool
        aez_temp_regime = np.where((aez_tclimate==4)&(aez_tzone==3)&(aez_temp_regime==0),6,aez_temp_regime) # Temperate, moderate
        aez_temp_regime = np.where((aez_tclimate==4)&(aez_tzone==4)&(aez_temp_regime==0),7,aez_temp_regime) # Temperate, cool
        aez_temp_regime = np.where((aez_tclimate>=2)&(aez_tclimate<=6)&(aez_tzone==5)&(permafrost>=3)&(aez_temp_regime==0),8,aez_temp_regime) # Boreal/Cold, no permafrost
        aez_temp_regime = np.where((aez_tclimate>=2)&(aez_tclimate<=6)&(aez_tzone==5)&(permafrost<=2)&(aez_temp_regime==0),9,aez_temp_regime) # Boreal/Cold, with permafrost
        aez_temp_regime = np.where((aez_tclimate>=2)&(aez_tclimate<=7)&(aez_tzone==6)&(aez_temp_regime==0),10,aez_temp_regime) # Arctic/Very Cold


        # 4th Step: Moisture Regime classes
        # Moisture Regime Class Definition
        # 1 = M1 (desert/arid areas, 0 <= LGP* < 60)
        # 2 = M2 (semi-arid/dry areas, 60 <= LGP* < 180)
        # 3 = M3 (sub-humid/moist areas, 180 <= LGP* < 270)
        # 4 = M4 (humid/wet areas, LGP* >= 270)
        aez_moisture_regime = np.zeros((self.im_height, self.im_width), dtype='int8')
        # check if LGP t>5 is greater or less than 330 days. If greater, LGP will be used; otherwise, LGP_equv will be used.
        aez_moisture_regime=np.where((lgpt_5>330)&(lgp>=270),4,aez_moisture_regime) # Class 4 (M4)
        aez_moisture_regime=np.where((lgpt_5>330)&(lgp>=180)&(lgp<270)&(aez_moisture_regime==0),3,aez_moisture_regime) # Class 3 (M3)
        aez_moisture_regime=np.where((lgpt_5>330)&(lgp>=60)&(lgp<180)&(aez_moisture_regime==0),2,aez_moisture_regime) # Class 2 (M2)
        aez_moisture_regime=np.where((lgpt_5>330)&(lgp>=0)&(lgp<60)&(aez_moisture_regime==0),1,aez_moisture_regime) # Class 1 (M1)
        aez_moisture_regime=np.where((lgpt_5<=330)&(lgp_equv>=270)&(aez_moisture_regime==0),4,aez_moisture_regime) # Class 4 (M4)
        aez_moisture_regime=np.where((lgpt_5<=330)&(lgp_equv>=180)&(lgp_equv<270)&(aez_moisture_regime==0),3,aez_moisture_regime) # Class 3 (M3)
        aez_moisture_regime=np.where((lgpt_5<=330)&(lgp_equv>=60)&(lgp_equv<180)&(aez_moisture_regime==0),2,aez_moisture_regime) # Class 2 (M2)
        aez_moisture_regime=np.where((lgpt_5<=330)&(lgp_equv>=0)&(lgp_equv<60)&(aez_moisture_regime==0),1,aez_moisture_regime) # Class 1 (M1)


        # Now, we will classify the agro-ecological zonation
        # By GAEZ v4 Documentation, there are prioritized sequential assignment of AEZ classes in order to ensure the consistency of classification
        if self.parallel:
            soil_terrain_lulc=soil_terrain_lulc.compute()

        self.soil_terrain_lulc=soil_terrain_lulc
        self.aez_moisture_regime=aez_moisture_regime
        self.aez_temp_regime=aez_temp_regime
        self.aez_tzone=aez_tzone
        self.aez_tclimate=aez_tclimate

        aez = np.zeros((self.im_height, self.im_width), dtype='int8')
        aez=np.where((soil_terrain_lulc==8)&(aez==0),56,aez) # if it's urban built-up lulc, Dominantly urban/built-up land
        aez=np.where((soil_terrain_lulc==7)&(aez==0),57,aez) # if it's water/ dominantly water
        aez=np.where((soil_terrain_lulc==1)&(aez==0),49,aez) # if it's dominantly very steep terrain/Dominantly very steep terrain 
        aez=np.where((soil_terrain_lulc==6)&(aez==0),51,aez) # if it's irrigated soils/ Land with ample irrigated soils
        aez=np.where((soil_terrain_lulc==2)&(aez==0),52,aez) # if it's hydromorphic soils/ Dominantly hydromorphic soils
        aez=np.where((aez_moisture_regime==1)&(aez==0),53,aez) # Desert/Arid climate
        aez=np.where((aez_temp_regime==9)&(aez_moisture_regime>=1)&(aez_moisture_regime<=4)&(aez==0),54,aez) # BO/Cold climate, with Permafrost
        aez=np.where((aez_temp_regime==10)&(aez_moisture_regime>=1)&(aez_moisture_regime<=4)&(aez==0),55,aez) # Arctic/ Very cold climate
        aez=np.where((soil_terrain_lulc==5)&(aez==0),50,aez) # Severe soil/terrain limitations
        #######
        aez=np.where((aez_temp_regime==1)&(aez_moisture_regime==2)&(soil_terrain_lulc==3)&(aez==0),1,aez) 
        aez=np.where((aez_temp_regime==1)&(aez_moisture_regime==2)&(soil_terrain_lulc==4)&(aez==0),2,aez) 
        aez=np.where((aez_temp_regime==1)&(aez_moisture_regime==3)&(soil_terrain_lulc==3)&(aez==0),3,aez) 
        aez=np.where((aez_temp_regime==1)&(aez_moisture_regime==3)&(soil_terrain_lulc==4)&(aez==0),4,aez) 
        aez=np.where((aez_temp_regime==1)&(aez_moisture_regime==4)&(soil_terrain_lulc==3)&(aez==0),5,aez) 
        aez=np.where((aez_temp_regime==1)&(aez_moisture_regime==4)&(soil_terrain_lulc==4)&(aez==0),6,aez) 
        #######
        aez=np.where((aez_temp_regime==2)&(aez_moisture_regime==2)&(soil_terrain_lulc==3)&(aez==0),7,aez) 
        aez=np.where((aez_temp_regime==2)&(aez_moisture_regime==2)&(soil_terrain_lulc==4)&(aez==0),8,aez) 
        aez=np.where((aez_temp_regime==2)&(aez_moisture_regime==3)&(soil_terrain_lulc==3)&(aez==0),9,aez) 
        aez=np.where((aez_temp_regime==2)&(aez_moisture_regime==3)&(soil_terrain_lulc==4)&(aez==0),10,aez) 
        aez=np.where((aez_temp_regime==2)&(aez_moisture_regime==4)&(soil_terrain_lulc==3)&(aez==0),11,aez) 
        aez=np.where((aez_temp_regime==2)&(aez_moisture_regime==4)&(soil_terrain_lulc==4)&(aez==0),12,aez) 
        #######
        aez=np.where((aez_temp_regime==3)&(aez_moisture_regime==2)&(soil_terrain_lulc==3)&(aez==0),13,aez) 
        aez=np.where((aez_temp_regime==3)&(aez_moisture_regime==2)&(soil_terrain_lulc==4)&(aez==0),14,aez) 
        aez=np.where((aez_temp_regime==3)&(aez_moisture_regime==3)&(soil_terrain_lulc==3)&(aez==0),15,aez) 
        aez=np.where((aez_temp_regime==3)&(aez_moisture_regime==3)&(soil_terrain_lulc==4)&(aez==0),16,aez) 
        aez=np.where((aez_temp_regime==3)&(aez_moisture_regime==4)&(soil_terrain_lulc==3)&(aez==0),17,aez) 
        aez=np.where((aez_temp_regime==3)&(aez_moisture_regime==4)&(soil_terrain_lulc==4)&(aez==0),18,aez)    
        #######
        aez=np.where((aez_temp_regime==4)&(aez_moisture_regime==2)&(soil_terrain_lulc==3)&(aez==0),19,aez) 
        aez=np.where((aez_temp_regime==4)&(aez_moisture_regime==2)&(soil_terrain_lulc==4)&(aez==0),20,aez) 
        aez=np.where((aez_temp_regime==4)&(aez_moisture_regime==3)&(soil_terrain_lulc==3)&(aez==0),21,aez) 
        aez=np.where((aez_temp_regime==4)&(aez_moisture_regime==3)&(soil_terrain_lulc==4)&(aez==0),22,aez) 
        aez=np.where((aez_temp_regime==4)&(aez_moisture_regime==4)&(soil_terrain_lulc==3)&(aez==0),23,aez) 
        aez=np.where((aez_temp_regime==4)&(aez_moisture_regime==4)&(soil_terrain_lulc==4)&(aez==0),24,aez)    
        #######
        aez=np.where((aez_temp_regime==5)&(aez_moisture_regime==2)&(soil_terrain_lulc==3)&(aez==0),25,aez) 
        aez=np.where((aez_temp_regime==5)&(aez_moisture_regime==2)&(soil_terrain_lulc==4)&(aez==0),26,aez) 
        aez=np.where((aez_temp_regime==5)&(aez_moisture_regime==3)&(soil_terrain_lulc==3)&(aez==0),27,aez) 
        aez=np.where((aez_temp_regime==5)&(aez_moisture_regime==3)&(soil_terrain_lulc==4)&(aez==0),28,aez) 
        aez=np.where((aez_temp_regime==5)&(aez_moisture_regime==4)&(soil_terrain_lulc==3)&(aez==0),29,aez) 
        aez=np.where((aez_temp_regime==5)&(aez_moisture_regime==4)&(soil_terrain_lulc==4)&(aez==0),30,aez)    
        #######
        aez=np.where((aez_temp_regime==6)&(aez_moisture_regime==2)&(soil_terrain_lulc==3)&(aez==0),31,aez) 
        aez=np.where((aez_temp_regime==6)&(aez_moisture_regime==2)&(soil_terrain_lulc==4)&(aez==0),32,aez) 
        aez=np.where((aez_temp_regime==6)&(aez_moisture_regime==3)&(soil_terrain_lulc==3)&(aez==0),33,aez) 
        aez=np.where((aez_temp_regime==6)&(aez_moisture_regime==3)&(soil_terrain_lulc==4)&(aez==0),34,aez) 
        aez=np.where((aez_temp_regime==6)&(aez_moisture_regime==4)&(soil_terrain_lulc==3)&(aez==0),35,aez) 
        aez=np.where((aez_temp_regime==6)&(aez_moisture_regime==4)&(soil_terrain_lulc==4)&(aez==0),36,aez)    
        #######  
        aez=np.where((aez_temp_regime==7)&(aez_moisture_regime==2)&(soil_terrain_lulc==3)&(aez==0),37,aez) 
        aez=np.where((aez_temp_regime==7)&(aez_moisture_regime==2)&(soil_terrain_lulc==4)&(aez==0),38,aez) 
        aez=np.where((aez_temp_regime==7)&(aez_moisture_regime==3)&(soil_terrain_lulc==3)&(aez==0),39,aez) 
        aez=np.where((aez_temp_regime==7)&(aez_moisture_regime==3)&(soil_terrain_lulc==4)&(aez==0),40,aez) 
        aez=np.where((aez_temp_regime==7)&(aez_moisture_regime==4)&(soil_terrain_lulc==3)&(aez==0),41,aez) 
        aez=np.where((aez_temp_regime==7)&(aez_moisture_regime==4)&(soil_terrain_lulc==4)&(aez==0),42,aez)    
        #######  
        aez=np.where((aez_temp_regime==8)&(aez_moisture_regime==2)&(soil_terrain_lulc==3)&(aez==0),43,aez) 
        aez=np.where((aez_temp_regime==8)&(aez_moisture_regime==2)&(soil_terrain_lulc==4)&(aez==0),44,aez) 
        aez=np.where((aez_temp_regime==8)&(aez_moisture_regime==3)&(soil_terrain_lulc==3)&(aez==0),45,aez) 
        aez=np.where((aez_temp_regime==8)&(aez_moisture_regime==3)&(soil_terrain_lulc==4)&(aez==0),46,aez) 
        aez=np.where((aez_temp_regime==8)&(aez_moisture_regime==4)&(soil_terrain_lulc==3)&(aez==0),47,aez) 
        aez=np.where((aez_temp_regime==8)&(aez_moisture_regime==4)&(soil_terrain_lulc==4)&(aez==0),48,aez)    
        #######   

        if self.set_mask:
            if self.parallel:
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask
            return np.where(mask, aez.astype('float32'), np.float32(np.nan))
        else:
            return aez                
        # #1st step: reclassifying the existing 12 classes of thermal climate into 6 major thermal climate.
        # # Class 1: Tropics, lowland
        # # Class 2: Tropics, highland
        # # Class 3: Subtropics
        # # Class 4: Temperate Climate
        # # Class 5: Boreal Climate
        # # Class 6: Arctic Climate
    
        # aez_tclimate = np.zeros((self.im_height, self.im_width), dtype=int)
        # # obj_utilities = UtilitiesCalc.UtilitiesCalc()

        # for i_r in range(self.im_height):
        #     for i_c in range(self.im_width):
        #         if self.set_mask:
        #             if self.im_mask[i_r, i_c] == self.nodata_val:
        #                 continue

        #             else:

        #                 # tropics highland
        #                 if tclimate[i_r, i_c] == 1:
        #                     aez_tclimate[i_r, i_c] = 1

        #                 elif tclimate[i_r, i_c] == 2:
        #                     aez_tclimate[i_r, i_c] = 2

        #                 elif tclimate[i_r, i_c] == 3:
        #                     aez_tclimate[i_r, i_c] = 3

        #                 elif tclimate[i_r, i_c] == 4:
        #                     aez_tclimate[i_r, i_c] = 3

        #                 elif tclimate[i_r, i_c] == 5:
        #                     aez_tclimate[i_r, i_c] = 3

        #                 # grouping all the temperate classes into a single class 4
        #                 elif tclimate[i_r, i_c] == 6:
        #                     aez_tclimate[i_r, i_c] = 4

        #                 elif tclimate[i_r, i_c] == 7:
        #                     aez_tclimate[i_r, i_c] = 4

        #                 elif tclimate[i_r, i_c] == 8:
        #                     aez_tclimate[i_r, i_c] = 4

        #                 # grouping all the boreal classes into a single class 5
        #                 elif tclimate[i_r, i_c] == 9:
        #                     aez_tclimate[i_r, i_c] = 5

        #                 elif tclimate[i_r, i_c] == 10:
        #                     aez_tclimate[i_r, i_c] = 5

        #                 elif tclimate[i_r, i_c] == 11:
        #                     aez_tclimate[i_r, i_c] = 5

        #                 # changing the arctic class into class 6
        #                 elif tclimate[i_r, i_c] == 12:
        #                     aez_tclimate[i_r, i_c] = 6

        # # 2nd Step: Classification of Thermal Zones
        # aez_tzone = np.zeros((self.im_height, self.im_width), dtype=int)


        # for i_r in range(self.im_height):
        #     for i_c in range(self.im_width):
        #         if self.set_mask:
        #             if self.im_mask[i_r, i_c] == self.nodata_val:
        #                 continue
        #             else:
        #                 mean_temp = np.copy(self.meanT_daily[i_r, i_c, :])
        #                 # meanT_monthly = obj_utilities.averageDailyToMonthly(mean_temp)
        #                 meanT_monthly=self.meanT_monthly
        #                 # one conditional parameter for temperature accumulation
        #                 temp_acc_10deg = np.copy(self.meanT_daily[i_r, i_c, :])
        #                 temp_acc_10deg[temp_acc_10deg < 10] = 0

        #                 # Warm Tzone (TZ1)
        #                 if np.sum(meanT_monthly >= 10) == 12 and np.mean(mean_temp) >= 20:
        #                     aez_tzone[i_r, i_c] = 1

        #                 # Moderately cool Tzone (TZ2)
        #                 elif np.sum(meanT_monthly >= 5) == 12 and np.sum(meanT_monthly >= 10) >= 8:
        #                     aez_tzone[i_r, i_c] = 2

        #                 # TZ3 Moderate
        #                 elif aez_tclimate[i_r, i_c] == 4 and np.sum(meanT_monthly >= 10) >= 5 and np.sum(mean_temp > 20) >= 75 and np.sum(temp_acc_10deg) > 3000:
        #                     aez_tzone[i_r, i_c] = 3

        #                 # TZ4 Cool
        #                 elif np.sum(meanT_monthly >= 10) >= 4 and np.mean(mean_temp) >= 0:
        #                     aez_tzone[i_r, i_c] = 4

        #                 # TZ5 Cold
        #                 elif np.sum(meanT_monthly >= 10) in range(1, 4) and np.mean(mean_temp) >= 0:
        #                     aez_tzone[i_r, i_c] = 5

        #                 # TZ6 Very cold
        #                 elif np.sum(meanT_monthly < 10) == 12 or np.mean(mean_temp) < 0:
        #                     aez_tzone[i_r, i_c] = 6

        # # 3rd Step: Creation of Temperature Regime Classes
        # # Temperature Regime Class Definition
        # # 1 = Tropics, lowland (TRC1)
        # # 2 = Tropics, highland (TRC2)
        # # 3 = Subtropics, warm (TRC3)
        # # 4 = Subtropics, moderately cool (TRC4)
        # # 5 = Subtropics, cool (TRC5)
        # # 6 = Temperate, moderate (TRC6)
        # # 7 = Temperate, cool (TRC7)
        # # 8 = Boreal, cold, no continuous or discontinuous occurrence of permafrost (TRC8)
        # # 9 = Boreal, cold, with continuous or discontinuous occurrence of permafrost (TRC9)
        # # 10 = Arctic, very cold (TRC10)

        # aez_temp_regime = np.zeros((self.im_height, self.im_width), dtype=int)

        # for i_r in range(self.im_height):
        #     for i_c in range(self.im_width):
        #         if self.set_mask:
        #             if self.im_mask[i_r, i_c] == self.nodata_val:
        #                 continue
        #             else:

        #                 if aez_tclimate[i_r, i_c] == 1 and aez_tzone[i_r, i_c] == 1:
        #                     aez_temp_regime[i_r, i_c] = 1  # Tropics, lowland

        #                 elif aez_tclimate[i_r, i_c] == 2 and aez_tzone[i_r, i_c] in [2, 4]:
        #                     aez_temp_regime[i_r, i_c] = 2  # Tropics, highland

        #                 elif aez_tclimate[i_r, i_c] == 3 and aez_tzone[i_r, i_c] == 1:
        #                     aez_temp_regime[i_r, i_c] = 3  # Subtropics, warm

        #                 elif aez_tclimate[i_r, i_c] == 3 and aez_tzone[i_r, i_c] == 2:
        #                     # Subtropics,moderate cool
        #                     aez_temp_regime[i_r, i_c] = 4

        #                 elif aez_tclimate[i_r, i_c] == 3 and aez_tzone[i_r, i_c] == 4:
        #                     aez_temp_regime[i_r, i_c] = 5  # Subtropics,cool

        #                 elif aez_tclimate[i_r, i_c] == 4 and aez_tzone[i_r, i_c] == 3:
        #                     # Temperate, moderate
        #                     aez_temp_regime[i_r, i_c] = 6

        #                 elif aez_tclimate[i_r, i_c] == 4 and aez_tzone[i_r, i_c] == 4:
        #                     aez_temp_regime[i_r, i_c] = 7  # Temperate, cool

        #                 elif aez_tclimate[i_r, i_c] in range(2, 6) and aez_tzone[i_r, i_c] == 5:
        #                     if np.logical_or(permafrost[i_r, i_c] == 1, permafrost[i_r, i_c] == 2) == False:
        #                         # Boreal/Cold, no
        #                         aez_temp_regime[i_r, i_c] = 8
        #                     else:
        #                         # Boreal/Cold, with permafrost
        #                         aez_temp_regime[i_r, i_c] = 9

        #                 elif aez_tclimate[i_r, i_c] in range(2, 7) and aez_tzone[i_r, i_c] == 6:
        #                     aez_temp_regime[i_r, i_c] = 10  # Arctic/Very Cold

        # # 4th Step: Moisture Regime classes
        # # Moisture Regime Class Definition
        # # 1 = M1 (desert/arid areas, 0 <= LGP* < 60)
        # # 2 = M2 (semi-arid/dry areas, 60 <= LGP* < 180)
        # # 3 = M3 (sub-humid/moist areas, 180 <= LGP* < 270)
        # # 4 = M4 (humid/wet areas, LGP* >= 270)

        # aez_moisture_regime = np.zeros(
        #     (self.im_height, self.im_width), dtype=int)

        # for i_r in range(self.im_height):
        #     for i_c in range(self.im_width):
        #         if self.set_mask:
        #             if self.im_mask[i_r, i_c] == self.nodata_val:
        #                 continue
        #             else:

        #                 # check if LGP t>5 is greater or less than 330 days. If greater, LGP will be used; otherwise, LGP_equv will be used.
        #                 if lgpt_5[i_r, i_c] > 330:

        #                     # Class 4 (M4)
        #                     if lgp[i_r, i_c] >= 270:
        #                         aez_moisture_regime[i_r, i_c] = 4

        #                     # Class 3 (M3)
        #                     elif lgp[i_r, i_c] >= 180 and lgp[i_r, i_c] < 270:
        #                         aez_moisture_regime[i_r, i_c] = 3

        #                     # Class 2 (M2)
        #                     elif lgp[i_r, i_c] >= 60 and lgp[i_r, i_c] < 180:
        #                         aez_moisture_regime[i_r, i_c] = 2

        #                     # Class 1 (M1)
        #                     elif lgp[i_r, i_c] >= 0 and lgp[i_r, i_c] < 60:
        #                         aez_moisture_regime[i_r, i_c] = 1

        #                 elif lgpt_5[i_r, i_c] <= 330:

        #                     # Class 4 (M4)
        #                     if lgp_equv[i_r, i_c] >= 270:
        #                         aez_moisture_regime[i_r, i_c] = 4

        #                     # Class 3 (M3)
        #                     elif lgp_equv[i_r, i_c] >= 180 and lgp_equv[i_r, i_c] < 270:
        #                         aez_moisture_regime[i_r, i_c] = 3

        #                     # Class 2 (M2)
        #                     elif lgp_equv[i_r, i_c] >= 60 and lgp_equv[i_r, i_c] < 180:
        #                         aez_moisture_regime[i_r, i_c] = 2

        #                     # Class 1 (M1)
        #                     elif lgp_equv[i_r, i_c] >= 0 and lgp_equv[i_r, i_c] < 60:
        #                         aez_moisture_regime[i_r, i_c] = 1

        # # Now, we will classify the agro-ecological zonation
        # # By GAEZ v4 Documentation, there are prioritized sequential assignment of AEZ classes in order to ensure the consistency of classification
        # aez = np.zeros((self.im_height, self.im_width), dtype=int)

        # for i_r in range(self.im_height):
        #     for i_c in range(self.im_width):
        #         if self.set_mask:
        #             if self.im_mask[i_r, i_c] == self.nodata_val:
        #                 continue
        #             else:
        #                 # if it's urban built-up lulc, Dominantly urban/built-up land
        #                 if soil_terrain_lulc[i_r, i_c] == 8:
        #                     aez[i_r, i_c] = 56

        #                 # if it's water/ dominantly water
        #                 elif soil_terrain_lulc[i_r, i_c] == 7:
        #                     aez[i_r, i_c] = 57

        #                 # if it's dominantly very steep terrain/Dominantly very steep terrain
        #                 elif soil_terrain_lulc[i_r, i_c] == 1:
        #                     aez[i_r, i_c] = 49

        #                 # if it's irrigated soils/ Land with ample irrigated soils
        #                 elif soil_terrain_lulc[i_r, i_c] == 6:
        #                     aez[i_r, i_c] = 51

        #                 # if it's hydromorphic soils/ Dominantly hydromorphic soils
        #                 elif soil_terrain_lulc[i_r, i_c] == 2:
        #                     aez[i_r, i_c] = 52

        #                 # Desert/Arid climate
        #                 elif aez_moisture_regime[i_r, i_c] == 1:
        #                     aez[i_r, i_c] = 53

        #                 # BO/Cold climate, with Permafrost
        #                 elif aez_temp_regime[i_r, i_c] == 9 and aez_moisture_regime[i_r, i_c] in [1, 2, 3, 4] == True:
        #                     aez[i_r, i_c] = 54

        #                 # Arctic/ Very cold climate
        #                 elif aez_temp_regime[i_r, i_c] == 10 and aez_moisture_regime[i_r, i_c] in [1, 2, 3, 4] == True:
        #                     aez[i_r, i_c] = 55

        #                 # Severe soil/terrain limitations
        #                 elif soil_terrain_lulc[i_r, i_c] == 5:
        #                     aez[i_r, i_c] = 50

        #                 #######
        #                 elif aez_temp_regime[i_r, i_c] == 1 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 1

        #                 elif aez_temp_regime[i_r, i_c] == 1 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 2

        #                 elif aez_temp_regime[i_r, i_c] == 1 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 3

        #                 elif aez_temp_regime[i_r, i_c] == 1 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 4

        #                 elif aez_temp_regime[i_r, i_c] == 1 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 5

        #                 elif aez_temp_regime[i_r, i_c] == 1 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 6
        #                 ####
        #                 elif aez_temp_regime[i_r, i_c] == 2 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 7

        #                 elif aez_temp_regime[i_r, i_c] == 2 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 8

        #                 elif aez_temp_regime[i_r, i_c] == 2 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 9

        #                 elif aez_temp_regime[i_r, i_c] == 2 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 10

        #                 elif aez_temp_regime[i_r, i_c] == 2 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 11

        #                 elif aez_temp_regime[i_r, i_c] == 2 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 12
        #                 ###
        #                 elif aez_temp_regime[i_r, i_c] == 3 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 13

        #                 elif aez_temp_regime[i_r, i_c] == 3 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 14

        #                 elif aez_temp_regime[i_r, i_c] == 3 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 15

        #                 elif aez_temp_regime[i_r, i_c] == 3 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 16

        #                 elif aez_temp_regime[i_r, i_c] == 3 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 17

        #                 elif aez_temp_regime[i_r, i_c] == 3 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 18
        #                 #####
        #                 elif aez_temp_regime[i_r, i_c] == 4 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 19

        #                 elif aez_temp_regime[i_r, i_c] == 4 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 20

        #                 elif aez_temp_regime[i_r, i_c] == 4 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 21

        #                 elif aez_temp_regime[i_r, i_c] == 4 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 22

        #                 elif aez_temp_regime[i_r, i_c] == 4 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 23

        #                 elif aez_temp_regime[i_r, i_c] == 4 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 24
        #                 #####
        #                 elif aez_temp_regime[i_r, i_c] == 5 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 25

        #                 elif aez_temp_regime[i_r, i_c] == 5 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 26

        #                 elif aez_temp_regime[i_r, i_c] == 5 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 27

        #                 elif aez_temp_regime[i_r, i_c] == 5 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 28

        #                 elif aez_temp_regime[i_r, i_c] == 5 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 29

        #                 elif aez_temp_regime[i_r, i_c] == 5 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 30
        #                 ######

        #                 elif aez_temp_regime[i_r, i_c] == 6 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 31

        #                 elif aez_temp_regime[i_r, i_c] == 6 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 32

        #                 elif aez_temp_regime[i_r, i_c] == 6 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 33

        #                 elif aez_temp_regime[i_r, i_c] == 6 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 34

        #                 elif aez_temp_regime[i_r, i_c] == 6 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 35

        #                 elif aez_temp_regime[i_r, i_c] == 6 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 36

        #                 ###
        #                 elif aez_temp_regime[i_r, i_c] == 7 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 37

        #                 elif aez_temp_regime[i_r, i_c] == 7 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 38

        #                 elif aez_temp_regime[i_r, i_c] == 7 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 39

        #                 elif aez_temp_regime[i_r, i_c] == 7 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 40

        #                 elif aez_temp_regime[i_r, i_c] == 7 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 41

        #                 elif aez_temp_regime[i_r, i_c] == 7 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 42
        #                 #####

        #                 elif aez_temp_regime[i_r, i_c] == 8 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 43

        #                 elif aez_temp_regime[i_r, i_c] == 8 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 44

        #                 elif aez_temp_regime[i_r, i_c] == 8 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 45

        #                 elif aez_temp_regime[i_r, i_c] == 8 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 46

        #                 elif aez_temp_regime[i_r, i_c] == 8 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
        #                     aez[i_r, i_c] = 47

        #                 elif aez_temp_regime[i_r, i_c] == 8 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
        #                     aez[i_r, i_c] = 48          

        # if self.set_mask:
        #     return np.where(self.im_mask, aez, np.nan)
        # else:        
        #     return aez
    
    """ 
    Note from Swun: In this code, the logic of temperature amplitude is not added 
    as it brings big discrepency in the temperature regime calculation (in India) 
    compared to previous code. However, the classification schema is now adjusted 
    according to Gunther's agreement and the documentation.
    """
         
    def getMultiCroppingZones(self, t_climate, lgp, lgp_t5, lgp_t10, ts_t0, ts_t10):
    # def getMultiCroppingZones(self, t_climate, lgp, lgp_t5, lgp_t10, ts_t10, ts_t0):
    # should it be ts_t0 or ts_t10 coming into this function?
        """
        This function refers to the assessment of multiple cropping potential
        across the area through matching both growth cycle and temperature
        requirements for individual suitable crops with time avaiability of 
        crop growth. The logic considers crop suitability for rainfed and 
        irrigated conditions.

        Args:
        ----------
        t_climate : a 2-D numpy array
            Thermal Climate.
        lgp : a 2-D numpy array
            Length of Growing Period.
        lgp_t5 : a 2-D numpy array
            Thermal growing period in days with mean daily temperatures above 5 degree Celsius.
        lgp_t10 : a 2-D numpy array
            Thermal growing period in days with mean daily temperatures above 10 degree Celsius.
        ts_t10 : a 2-D numpy array
            Accumulated temperature (degree-days) on days when mean daily temperature is greater or equal to 10 degree Celsius.
        ts_t0 : a 2-D numpy array
            Accumulated temperature (degree-days) on days when mean daily temperature is greater or equal to 0 degree Celsius.
        tsg_t5 : a 2-D numpy array
            Accumulated temperature on growing period days when mean daily temperature is greater or equal to 5 degree Celsius.
        tsg_t10 : a 2-D numpy array
            Accumulated temperature on growing period days when mean daily temperature is greater or equal to 10 degree Celsius.

        Returns
        -------
        A list of two 2-D numpy arrays. The first array refers to multi-cropping
        zone for rainfed condition, and the second refers to multi-cropping zone
        for irrigated condition.

        """    
        if self.parallel:
            import dask.array as da
            interp_daily_np=UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D).smoothDailyTemp(self.doy_start-1,self.doy_end-1, self.im_mask, self.meanT_daily)
            interp_daily_da=da.from_array(interp_daily_np,chunks=self.chunk3D)
            # interp_daily_da=da.from_array(self.interp_daily_temp,chunks=self.chunk3D)
            interp_meanT_veg_T5=np.where(interp_daily_da>=5.,interp_daily_da,np.nan)   
            interp_meanT_veg_T10=np.where(interp_daily_da>=10.,interp_daily_da,np.nan)   
            ts_g_t5=np.nansum(interp_meanT_veg_T5,axis=2).compute()   
            ts_g_t10=np.nansum(interp_meanT_veg_T10,axis=2).compute()   
        else:
            interp_meanT_veg_T5=np.where(self.interp_daily_temp>=5.,self.interp_daily_temp,np.float32(np.nan))   
            interp_meanT_veg_T10=np.where(self.interp_daily_temp>=10.,self.interp_daily_temp,np.float32(np.nan))   
            ts_g_t5=np.nansum(interp_meanT_veg_T5,axis=2)
            ts_g_t10=np.nansum(interp_meanT_veg_T10,axis=2)

        del interp_meanT_veg_T5, interp_meanT_veg_T10          

        # defining the constant arrays for rainfed and irrigated conditions, all pixel values start with 1
        multi_crop_rain = np.ones((self.im_height, self.im_width), dtype = 'int8') # all values started with Zone A   
        multi_crop_irr = np.ones((self.im_height, self.im_width), dtype = 'int8') # all vauels starts with Zone A   
              
        """Multi cropping zonation for rainfed conditions"""
        multi_crop_rain=np.where((t_climate==1)&(lgp>=360)&(lgp_t5>=360)&(lgp_t10>=360)&(ts_t0>=7200)&(ts_t10>=7000),8,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate==1)&(lgp>=300)&(lgp_t5>=300)&(lgp_t10>=240)&(ts_t0>=7200)&(ts_g_t5>=5100)&(ts_g_t10>=4800)&(multi_crop_rain==1),6,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate==1)&(lgp>=270)&(lgp_t5>=270)&(lgp_t10>=165)&(ts_t0>=5500)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_rain==1),4,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate==1)&(lgp>=240)&(lgp_t5>=240)&(lgp_t10>=165)&(ts_t0>=6400)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_rain==1),4,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate==1)&(lgp>=210)&(lgp_t5>=240)&(lgp_t10>=165)&(ts_t0>=7200)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_rain==1),4,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate==1)&(lgp>=220)&(lgp_t5>=220)&(lgp_t10>=120)&(ts_t0>=5500)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_rain==1),3,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate==1)&(lgp>=200)&(lgp_t5>=200)&(lgp_t10>=120)&(ts_t0>=6400)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_rain==1),3,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate==1)&(lgp>=180)&(lgp_t5>=200)&(lgp_t10>=120)&(ts_t0>=7200)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_rain==1),3,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate==1)&(lgp>=45)&(lgp_t5>=120)&(lgp_t10>=90)&(ts_t0>=1600)&(ts_t10>=1200)&(multi_crop_rain==1),2,multi_crop_rain)    

        multi_crop_rain=np.where((t_climate!=1)&(lgp>=360)&(lgp_t5>=360)&(lgp_t10>=330)&(ts_t0>=7200)&(ts_t10>=7000)&(multi_crop_rain==1),8,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate!=1)&(lgp>=330)&(lgp_t5>=330)&(lgp_t10>=270)&(ts_t0>=5700)&(ts_t10>=5500)&(multi_crop_rain==1),7,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate!=1)&(lgp>=300)&(lgp_t5>=300)&(lgp_t10>=240)&(ts_t0>=5400)&(ts_t10>=5100)&(ts_g_t5>=5100)&(ts_g_t10>=4800)&(multi_crop_rain==1),6,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate!=1)&(lgp>=240)&(lgp_t5>=270)&(lgp_t10>=180)&(ts_t0>=4800)&(ts_t10>=4500)&(ts_g_t5>=4300)&(ts_g_t10>=4000)&(multi_crop_rain==1),5,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate!=1)&(lgp>=210)&(lgp_t5>=240)&(lgp_t10>=165)&(ts_t0>=4500)&(ts_t10>=3600)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_rain==1),4,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate!=1)&(lgp>=180)&(lgp_t5>=200)&(lgp_t10>=120)&(ts_t0>=3600)&(ts_t10>=3000)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_rain==1),3,multi_crop_rain)   
        multi_crop_rain=np.where((t_climate!=1)&(lgp>=45)&(lgp_t5>=120)&(lgp_t10>=90)&(ts_t0>=1600)&(ts_t10>=1200)&(multi_crop_rain==1),2,multi_crop_rain)                               
        
        """Multi cropping zonation for irrigated conditions"""
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=360)&(lgp_t10>=360)&(ts_t0>=7200)&(ts_t10>=7000),8,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=300)&(lgp_t10>=240)&(ts_t0>=7200)&(ts_g_t5>=5100)&(ts_g_t10>=4800)&(multi_crop_irr==1),6,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=270)&(lgp_t10>=165)&(ts_t0>=5500)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_irr==1),4,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=240)&(lgp_t10>=165)&(ts_t0>=6400)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_irr==1),4,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=240)&(lgp_t10>=165)&(ts_t0>=7200)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_irr==1),4,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=220)&(lgp_t10>=120)&(ts_t0>=5500)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_irr==1),3,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=200)&(lgp_t10>=120)&(ts_t0>=6400)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_irr==1),3,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=200)&(lgp_t10>=120)&(ts_t0>=7200)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_irr==1),3,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=120)&(lgp_t10>=90)&(ts_t0>=1600)&(ts_t10>=1200)&(multi_crop_irr==1),2,multi_crop_irr)   

        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=360)&(lgp_t10>=330)&(ts_t0>=7200)&(ts_t10>=7000)&(multi_crop_irr==1),8,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=330)&(lgp_t10>=270)&(ts_t0>=5700)&(ts_t10>=5500)&(multi_crop_irr==1),7,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=300)&(lgp_t10>=240)&(ts_t0>=5400)&(ts_t10>=5100)&(ts_g_t5>=5100)&(ts_g_t10>=4800)&(multi_crop_irr==1),6,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=270)&(lgp_t10>=180)&(ts_t0>=4800)&(ts_t10>=4500)&(ts_g_t5>=4300)&(ts_g_t10>=4000)&(multi_crop_irr==1),5,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=240)&(lgp_t10>=165)&(ts_t0>=4500)&(ts_t10>=3600)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_irr==1),4,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=200)&(lgp_t10>=120)&(ts_t0>=3600)&(ts_t10>=3000)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_irr==1),3,multi_crop_irr)   
        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=120)&(lgp_t10>=90)&(ts_t0>=1600)&(ts_t10>=1200)&(multi_crop_irr==1),2,multi_crop_irr)   

        if self.set_mask:
            if self.parallel:
                # print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask 
            return [np.where(mask, multi_crop_rain.astype('float32'), np.float32(np.nan)), np.where(mask, multi_crop_irr.astype('float32'), np.float32(np.nan))]   
        else:        
            return [multi_crop_rain, multi_crop_irr]   
                        
    

#----------------- End of file -------------------------#