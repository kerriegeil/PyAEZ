"""
PyAEZ version 2.1.0 (June 2023)
This ClimateRegime Class read/load and calculates the agro-climatic indicators
required to run PyAEZ.  
2021: N. Lakmal Deshapriya
2022/2023: Swun Wunna Htet and Kittiphon Boonma

"""

import numpy as np
# from pyaez import UtilitiesCalc, ETOCalc, LGPCalc
import UtilitiesCalc_test as UtilitiesCalc #KLG
import ETOCalc_test as ETOCalc # KLG
import LGPCalc_loopchunks as LGPCalc # KLG
from collections import OrderedDict
import psutil
import dask.array as da
import dask
import warnings
from time import time as timer
from dask.distributed import Client,LocalCluster



# np.seterr(divide='ignore', invalid='ignore') # ignore "divide by zero" or "divide by NaN" warning

# Initiate ClimateRegime Class instance
class ClimateRegime(object):
    def setParallel(self,var3D,parallel=False,nchunks=None,reduce_mem_used=False):
        # if ~nchunks:
        #     nchunks=UtilitiesCalc.UtilitiesCalc().getChunks()

        if parallel:
            self.parallel=True

            # we parallelize by chunking longitudes
            self.chunk2D,self.chunk3D,self.chunksize3D_MB,self.nchunks=UtilitiesCalc.UtilitiesCalc().setChunks(nchunks,var3D.shape,reduce_mem_used)
            # =var3D.nbytes/1E6/nchunks   # allow user to see the chunksize in MB   
            # self.nchunks=nchunks        
        else:
            self.parallel=False
            self.chunk3D=None
            self.chunk2D=None
            self.chunksize3D_MB=None
            self.nchunks=None
        
    def setLocationTerrainData(self, lat_min, lat_max, location, elevation):  #KLG
    # def setLocationTerrainData(self, lat_min, lat_max, elevation): 
    # def setLocationTerrainData(self, lats, elevation):  #option to take all lats as an input #KLG
        # why not take in an array of all latitudes here instead of regenerating lats from min/max  #KLG
        # could avoid slight shifts/precision problems with the grid  #KLG
        """Load geographical extents and elevation data in to the Class, 
           and create a latitude map

        Args:
            lat_min (float): the minimum latitude of the AOI in decimal degrees
            lat_max (float): the maximum latitude of the AOI in decimal degrees
            elevation (2D NumPy): elevation map in metres
        """
        if self.parallel:
            self.elevation = elevation.rechunk(chunks=self.chunk2D)
        else:        
            self.elevation = elevation
        self.im_height = elevation.shape[0]
        self.im_width = elevation.shape[1]
        # self.latitude = UtilitiesCalc.UtilitiesCalc().generateLatitudeMap(lat_min, lat_max, self.im_height, self.im_width) 
        self.latitude = UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D).generateLatitudeMap(lat_min, lat_max, location, self.im_height, self.im_width)  #KLG
        # self.latitude = UtilitiesCalc.UtilitiesCalc().generateLatitudeMap(lats, location)  #option to take all lats as an input  #KLG
        
    
    def setStudyAreaMask(self, admin_mask, no_data_value):
        """Set clipping mask of the area of interest (optional)

        Args:
            admin_mask (2D NumPy/Binary): mask to extract only region of interest
            no_data_value (int): pixels with this value will be omitted during PyAEZ calculations
        """    
        self.im_mask = admin_mask.rechunk(chunks=self.chunk2D).astype('int8')
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
        self.doy_start=1  #KLG
        self.doy_end=min_temp.shape[2]  #KLG
        
        rel_humidity[rel_humidity > 0.99] = 0.99
        rel_humidity[rel_humidity < 0.05] = 0.05
        short_rad[short_rad < 0] = 0
        wind_speed[wind_speed < 0] = 0

        # self.meanT_daily = np.zeros((self.im_height, self.im_width, 365))  
        # self.totalPrec_daily = np.zeros((self.im_height, self.im_width, 365))  
        # self.pet_daily = np.zeros((self.im_height, self.im_width, 365))  
        # self.minT_daily = np.zeros((self.im_height, self.im_width, 365))  
        # self.maxT_daily = np.zeros((self.im_height, self.im_width, 365))  


        # Interpolate monthly to daily data
        obj_utilities = UtilitiesCalc.UtilitiesCalc()

        meanT_monthly = (min_temp+max_temp)/2

        # for i_row in range(self.im_height):  
        #     for i_col in range(self.im_width):  

        #         if self.set_mask:  
        #             if self.im_mask[i_row, i_col] == self.nodata_val:  
        #                 continue  

        #         self.meanT_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(meanT_monthly[i_row, i_col,:], 1, 365)  
        #         self.totalPrec_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(precipitation[i_row, i_col,:], 1, 365, no_minus_values=True)  
        #         self.minT_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(min_temp[i_row, i_col,:], 1, 365)  
        #         self.maxT_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(max_temp[i_row, i_col,:], 1, 365)  
        #         radiation_daily = obj_utilities.interpMonthlyToDaily(short_rad[i_row, i_col,:], 1, 365, no_minus_values=True)  
        #         wind_daily = obj_utilities.interpMonthlyToDaily(wind_speed[i_row, i_col,:], 1, 365, no_minus_values=True)  
        #         rel_humidity_daily = obj_utilities.interpMonthlyToDaily(rel_humidity[i_row, i_col,:], 1, 365, no_minus_values=True)  

        #         # calculation of reference evapotranspiration (ETo) 
        #         obj_eto = ETOCalc.ETOCalc(1, 365, self.latitude[i_row, i_col], self.elevation[i_row, i_col])  
        #         shortrad_daily_MJm2day = (radiation_daily*3600*24)/1000000 # convert w/m2 to MJ/m2/day  
        #         obj_eto.setClimateData(self.minT_daily[i_row, i_col, :], self.maxT_daily[i_row, i_col, :], wind_daily, shortrad_daily_MJm2day, rel_humidity_daily) 
        #         self.pet_daily[i_row, i_col, :] = obj_eto.calculateETO()  
                
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
        # cluster=LocalCluster(n_workers=1, threads_per_worker=20)
        # client=Client(cluster)

        self.doy_start=1  #KLG
        self.doy_end=min_temp.shape[2]  #KLG

        # if self.set_mask:
        #     if self.parallel:
        #         mask_monthly=np.tile(self.im_mask.compute()[:,:,np.newaxis],(1,1,12)) # mask with matching dims
        #     else:
        #         mask_monthly=np.tile(self.im_mask[:,:,np.newaxis],(1,1,12)) # mask with matching dims      

        if self.parallel:
            min_temp=min_temp.rechunk(chunks=self.chunk3D)
            max_temp=max_temp.rechunk(chunks=self.chunk3D)
            tmn_delay=min_temp.to_delayed().ravel()
            tmx_delay=max_temp.to_delayed().ravel()

            short_rad=short_rad.rechunk(chunks=self.chunk3D) # chunk
            short_rad=da.where(short_rad < 0, 0, short_rad)  # elim negatives
            short_rad = short_rad*0.0864 #(3600.*24./1000000.)       # convert units
            srad_delay=short_rad.to_delayed().ravel()
            # del short_rad   

            wind_speed=wind_speed.rechunk(chunks=self.chunk3D)     # chunk
            wind_speed=da.where(wind_speed < 0.5, 0.5, wind_speed) # elim negative and small values
            wind_delay=wind_speed.to_delayed().ravel() 

            rel_humidity=rel_humidity.rechunk(chunks=self.chunk3D)        # chunk
            rel_humidity=da.where(rel_humidity > 0.99, 0.99,rel_humidity) # elim high values
            rel_humidity=da.where(rel_humidity < 0.05, 0.05,rel_humidity) # elim low values
            rh_delay=rel_humidity.to_delayed().ravel()
            # del rel_humidity  

            lat_delay=self.latitude.to_delayed().ravel()
            elev_delay=self.elevation.to_delayed().ravel()  

            ### CALCULATE PET_DAILY ###
            # print('in ClimateRegime, computing chunk_shapes in parallel')
            # chunk_shapes=dask.compute([chunk.shape for chunk in tmn_delay])
            # zipvars=zip(chunk_shapes[0][:],lat_delay,elev_delay,tmn_delay,tmx_delay,wind_delay,srad_delay,rh_delay)
            zipvars=zip(lat_delay,elev_delay,tmn_delay,tmx_delay,wind_delay,srad_delay,rh_delay)
            obj_eto=ETOCalc.ETOCalc() # less copying of variables to save RAM
            # task_list=[dask.delayed(obj_eto.calculateETO)(self.doy_start,self.doy_end,cshape,lat,el,tmn,tmx,u,srad,rh) for cshape,lat,el,tmn,tmx,u,srad,rh in zipvars]
            task_list=[dask.delayed(obj_eto.calculateETO)(self.doy_start,self.doy_end,lat,el,tmn,tmx,u,srad,rh) for lat,el,tmn,tmx,u,srad,rh in zipvars]
            print('in ClimateRegime, computing pet_daily in parallel')
            result_chunks=dask.compute(*task_list)
            self.pet_daily=np.concatenate(result_chunks,axis=1) 
            # print(psutil.virtual_memory().free/1E9)
            # self.pet_daily=da.concatenate(result_chunks,axis=1)                                        
            # print(psutil.virtual_memory().free/1E9)
            del result_chunks,tmn_delay,tmx_delay,srad_delay,short_rad,wind_delay,wind_speed,rh_delay,rel_humidity,lat_delay,elev_delay,zipvars,task_list
            # print(psutil.virtual_memory().free/1E9)
        else:
            rel_humidity[rel_humidity > 0.99] = 0.99
            rel_humidity[rel_humidity < 0.05] = 0.05
            short_rad[short_rad < 0] = 0
            short_rad = short_rad*0.0864 #3600.*24./1000000.
            # wind_speed[wind_speed < 0] = 0
            wind_speed[wind_speed < 0.5] = 0.5
            
            obj_eto=ETOCalc.ETOCalc()
            self.pet_daily= obj_eto.calculateETO(self.doy_start,self.doy_end,min_temp.shape,self.latitude,self.elevation,min_temp,max_temp,wind_speed,short_rad,rel_humidity)

        ### CALCULATE MEANT_DAILY ###
        self.meanT_daily = 0.5*(min_temp + max_temp)  #KLG
        
        obj_utilities = UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D)  #KLG

        # sea level temperature
        # P over PET ratio (to eliminate nan in the result, nan is replaced with zero)
        if self.parallel:
            ### CALCULATE MEANT_MONTHLY_SEALEVEL
            # we only ever use monthly mean sea level so it's pointless to carry daily data in RAM
            # print(self.im_height,self.im_width,self.doy_end,self.chunk3D)
            meanT_daily_sealevel = self.meanT_daily + (da.broadcast_to(self.elevation.compute()[:,:,np.newaxis],(self.im_height,self.im_width,self.doy_end)).rechunk(chunks=self.chunk3D)/100*0.55)
            print('in ClimateRegime, agg daily to meanT_monthly_sealevel in parallel')
            self.meanT_monthly_sealevel = obj_utilities.averageDailyToMonthly(meanT_daily_sealevel)  #KLG
            del meanT_daily_sealevel
            if self.set_mask:
                mask_monthly=np.broadcast_to(self.im_mask.compute()[:,:,np.newaxis],(self.im_height,self.im_width,12))
                self.meanT_monthly_sealevel = np.where(mask_monthly,np.float32(self.meanT_monthly_sealevel),np.float32(np.nan))
            
            ### CALCULATE P_BY_PET_MONTHLY ####
            # same for P_by_PET_monthly, we only use monthly
            # delay input data chunks
            pr_delayed=precipitation.rechunk(chunks=self.chunk3D).to_delayed().ravel()
            pet_delayed=da.from_array(self.pet_daily,chunks=self.chunk3D).to_delayed().ravel()
            mask_delayed=da.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,12),chunks=(self.chunk2D[0],self.chunk2D[1],12)).to_delayed().ravel()

            # print(precipitation.rechunk(chunks=self.chunk3D))
            # print(da.from_array(self.pet_daily,chunks=self.chunk3D))
            # print(da.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,12),chunks=(self.chunk2D[0],self.chunk2D[1],12)))

            def calc_P_by_PET_monthly(pet,pr,mask,obj_util):
                with np.errstate(divide='ignore', invalid='ignore'):
                    pet=np.where(pet==0,np.float32(np.nan),np.float32(pet)) # replace inf in the P_by result with nan
                    P_by_PET_daily = pr/pet # daily values
                    # print('sending to Utilities, agg P_by_PET daily to monthly in parallel')
                    P_by_PET_monthly = obj_util.averageDailyToMonthly(P_by_PET_daily)  # monthly values
                    # mask=da.broadcast_to(mask[:,:,np.newaxis],(mask.shape[0],mask.shape[1],12)) # mask with matching dims
                    P_by_PET_monthly = np.where(mask,np.float32(P_by_PET_monthly),np.float32(np.nan)) # implement mask
                return P_by_PET_monthly
            
            # compute in parallel on chunks
            task_list=[dask.delayed(calc_P_by_PET_monthly)(pet_chunk,pr_chunk,mask_chunk,obj_utilities) for pet_chunk,pr_chunk,mask_chunk in zip(pr_delayed,pet_delayed,mask_delayed)]
            print('in ClimateRegime, computing P_by_PET in parallel')
            results=dask.compute(*task_list)
            # concatenate chunks
            self.P_by_PET_monthly=np.concatenate(results,axis=1) 
            del results,task_list,pr_delayed,pet_delayed,mask_delayed

            # pr=precipitation.rechunk(chunks=self.chunk3D)
            # pet=da.from_array(self.pet_daily,chunks=self.chunk3D)
            # mask_3D=da.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,12),chunks=(self.chunk2D[0],self.chunk2D[1],12))
            # with np.errstate(divide='ignore', invalid='ignore'):
            #     pet=da.where(pet==0,np.float32(np.nan),np.float32(pet))
            #     P_by_PET_daily = pr/pet
            #     print('sending to Utilities, agg P_by_PET daily to monthly in parallel')
            #     P_by_PET_monthly = obj_utilities.averageDailyToMonthly(P_by_PET_daily)
            #     P_by_PET_monthly = da.where(mask_3D,np.float32(P_by_PET_monthly),np.float32(np.nan))
            # self.P_by_PET_monthly=P_by_PET_monthly.compute()

            # del mask_3D,pet,P_by_PET_daily,P_by_PET_monthly            

            # nanzero_mask=da.where((self.pet_daily==0)|(~np.isfinite(self.pet_daily)),0,1).astype('int')  # find where pet_daily equals zero or nan, these location should be nan in P_by_PET_daily
            # pet_daily=np.where(nanzero_mask,self.pet_daily,1)  # local variable with no zero or nan
            # self.P_by_PET_daily = precipitation / pet_daily  # division without zero or nan warnings
            # self.P_by_PET_daily = da.where(nanzero_mask,self.P_by_PET_daily,np.nan).astype('float32') # put the nans back
            # print(psutil.virtual_memory().free/1E9)
            # del nanzero_mask,pet_daily
        else:
            ###### FIX THIS PART #######
            meanT_daily_sealevel = self.meanT_daily + np.expand_dims(self.elevation/100*0.55,axis=2) # automatic broadcasting #KLG   
            self.meanT_monthly_sealevel = obj_utilities.averageDailyToMonthly(meanT_daily_sealevel)  #KLG
            del meanT_daily_sealevel
            if self.set_mask:
                mask_monthly=np.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,12))
                self.meanT_monthly_sealevel = np.where(mask_monthly,np.float32(self.meanT_monthly_sealevel),np.float32(np.nan))            

            with np.errstate(invalid='ignore',divide='ignore'):
                self.P_by_PET_daily = np.nan_to_num(precipitation / self.pet_daily)

        self.set_monthly = False

        # # smoothed daily temperature
        # # create a mask of all 1's if the user doesn't provide a mask
        # if self.set_mask:
        #     if self.parallel:
        #         mask=self.im_mask.compute()
        #     else:
        #         mask=self.im_mask
        # else:
        #     mask=np.ones((self.im_height,self.im_width),dtype='int')

        # # print(psutil.virtual_memory().free/1E9)
        # obj_utilities = UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D)  #KLG
        # self.interp_daily_temp=obj_utilities.smoothDailyTemp(self.doy_start,self.doy_end, mask, self.meanT_daily)

        ### SET DAILY VARIABLES TO CLASS OBJECT ###
        self.maxT_daily = max_temp
        self.totalPrec_daily = precipitation.rechunk(chunks=self.chunk3D)  #KLG
        # del precipitation

        ### CALCULATE MONTHLY AND ANNUAL MEANS ###
        # adding other small things to RAM so save compute time later
        # monthly mean T
        # obj_utilities = UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D)  #KLG   
        print('in ClimateRegime, computing meanT_monthly, totalPrec_monthly')     
        self.meanT_monthly = obj_utilities.averageDailyToMonthly(self.meanT_daily)
        # monthly mean precip
        self.totalPrec_monthly = obj_utilities.averageDailyToMonthly(self.totalPrec_daily) 
        if self.set_mask:
            if self.parallel:
                mask_monthly=np.broadcast_to(self.im_mask.compute()[:,:,np.newaxis],(self.im_height,self.im_width,12))
                # mask=np.tile(self.im_mask.compute()[:,:,np.newaxis],(1,1,12)) # mask with matching dims
            else:
                # mask=np.tile(self.im_mask[:,:,np.newaxis],(1,1,12)) # mask with matching dims
                mask_monthly=np.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,12))

            self.meanT_monthly=np.where(mask_monthly,np.float32(self.meanT_monthly),np.float32(np.nan))
            self.totalPrec_monthly=np.where(mask_monthly,np.float32(self.totalPrec_monthly),np.float32(np.nan))
        
        # annual mean T
        if self.parallel:
            print('in ClimateRegime, computing annual_Tmean')
            self.annual_Tmean = da.mean(self.meanT_daily, axis = 2).compute()
            if self.set_mask:
                self.annual_Tmean=np.where(self.im_mask.compute(),np.float32(self.annual_Tmean),np.float32(np.nan))            
        else:
            self.annual_Tmean = np.mean(self.meanT_daily, axis = 2)
            if self.set_mask:
                self.annual_Tmean=np.where(self.im_mask,np.float32(self.annual_Tmean),np.float32(np.nan))            

        # annual mean T
        if self.parallel:
            print('in ClimateRegime, computing annual_accPrec')
            self.annual_accPrec = da.sum(self.totalPrec_daily, axis = 2).compute()
            if self.set_mask:
                self.annual_accPrec=np.where(self.im_mask.compute(),np.float32(self.annual_accPrec),np.float32(np.nan))               
        else:
            self.annual_accPrec = np.sum(self.totalPrec_daily, axis = 2) 
            if self.set_mask:
                self.annual_accPrec=np.where(self.im_mask,np.float32(self.annual_accPrec),np.float32(np.nan))                                
        # print(psutil.virtual_memory().free/1E9)        


    def getThermalClimate(self):
        """Classification of rainfall and temperature seasonality into thermal climate classes

        Returns:
            2D NumPy: Thermal Climate classification
        """        
        # Note that currently, this thermal climate is designed only for the northern hemisphere, southern hemisphere is not implemented yet.
        # if self.chunk3D:
        #     thermal_climate = da.empty((self.im_height,self.im_width),dtype='float32',chunks=self.chunk2D)  #KLG

        # else:
        print('initializing thermal_climate')
        thermal_climate = np.empty((self.im_height,self.im_width),dtype='float32')  #KLG
        thermal_climate[:] = np.nan  #KLG

        print('converting to monthly')
        # converting daily to monthly  #KLG
        # everything returned in memory as numpy arrays
        # obj_utilities = UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D)  #KLG
        # print('computing meanT_monthly_sealevel')
        # meanT_monthly_sealevel = obj_utilities.averageDailyToMonthly(self.meanT_daily_sealevel)  #KLG
        meanT_monthly_sealevel=self.meanT_monthly_sealevel
        # print('computing meanT_monthly')
        # meanT_monthly = obj_utilities.averageDailyToMonthly(self.meanT_daily)  #KLG
        meanT_monthly=self.meanT_monthly        
        # print('computing P_by_PET_monthly')
        # P_by_PET_monthly = obj_utilities.averageDailyToMonthly(self.P_by_PET_daily)  #KLG
        P_by_PET_monthly=self.P_by_PET_monthly

        print('preparing .where inputs')
        # things we need to assign thermal_climate values  #KLG
        # compute them here for readability below  #KLG
        summer_PET0=P_by_PET_monthly[:,:,3:9].sum(axis=2) # Apr-Sep  #KLG
        JFMSON=[0,1,2,9,10,11]  #KLG
        winter_PET0=P_by_PET_monthly[:,:,JFMSON].sum(axis=2) # Oct-Mar  #KLG

        min_sealev_meanT=meanT_monthly_sealevel.min(axis=2)  #KLG
        Ta_diff=meanT_monthly_sealevel.max(axis=2) - meanT_monthly_sealevel.min(axis=2)  #KLG
        meanT=meanT_monthly.mean(axis=2)  #KLG
        nmo_ge_10C=(meanT_monthly_sealevel >= 10).sum(axis=2)  #KLG

        prsum=self.annual_accPrec
        
        print('computing monthly pr')
        if self.chunk3D:
            # print('in ClimateRegime, computing prsum in parallel')
            # prsum=self.totalPrec_daily.sum(axis=2).compute()  #KLG
            print('in ClimateRegime, computing latitude in parallel')
            latitude=self.latitude.compute()
        else:
            # prsum=self.totalPrec_daily.sum(axis=2)  #KLG
            latitude=self.latitude

        # print(thermal_climate)
        # print(meanT_monthly_sealevel)
        # print(meanT_monthly)
        # print(P_by_PET_monthly)
        # print(prsum)

        # print(summer_PET0)
        # print(winter_PET0)
        # print(min_sealev_meanT)
        # print(Ta_diff)
        # print(meanT)
        # print(nmo_ge_10C)

        print('categorizing pixels')
        # assign values  #KLG
        # use the nan initialization to make sure we don't overwrite and previously assigned values  #KLG
        # Tropics  #KLG
        # Tropical lowland  #KLG
        thermal_climate=np.where((min_sealev_meanT>=18.) & (Ta_diff<15.) & (meanT>=20.),np.float32(1),np.float32(thermal_climate))  #KLG
        # Tropical highland  #KLG
        thermal_climate=np.where((min_sealev_meanT>=18.) & (Ta_diff<15.) & (meanT<20.) & ~np.isfinite(thermal_climate),np.float32(2),np.float32(thermal_climate))  #KLG
        
        # SubTropic  #KLG
        # Subtropics Low Rainfall  #KLG
        thermal_climate=np.where((min_sealev_meanT>=5.) & (nmo_ge_10C>=8) & (prsum<250) & ~np.isfinite(thermal_climate),np.float32(3),np.float32(thermal_climate))  #KLG
        # Subtropics Summer Rainfall  #KLG
        thermal_climate=np.where((min_sealev_meanT>=5.) & (nmo_ge_10C>=8) & (latitude>=0) & (summer_PET0>=winter_PET0) & ~np.isfinite(thermal_climate),np.float32(4),np.float32(thermal_climate))  #KLG
        thermal_climate=np.where((min_sealev_meanT>=5.) & (nmo_ge_10C>=8) & (latitude<0) & (summer_PET0<winter_PET0) & ~np.isfinite(thermal_climate),np.float32(4),np.float32(thermal_climate))  #KLG
        # Subtropics Winter Rainfall  #KLG
        thermal_climate=np.where((min_sealev_meanT>=5.) & (nmo_ge_10C>=8) & (latitude>=0) & (summer_PET0<winter_PET0) & ~np.isfinite(thermal_climate),np.float32(5),np.float32(thermal_climate))  #KLG
        thermal_climate=np.where((min_sealev_meanT>=5.) & (nmo_ge_10C>=8) & (latitude<0) & (summer_PET0>=winter_PET0) & ~np.isfinite(thermal_climate),np.float32(5),np.float32(thermal_climate))  #KLG
        
        # Temperate  #KLG
        # Oceanic Temperate  #KLG
        thermal_climate=np.where((nmo_ge_10C>=4) & (Ta_diff<=20) & ~np.isfinite(thermal_climate),np.float32(6),np.float32(thermal_climate))  #KLG
        # Sub-Continental Temperate  #KLG
        thermal_climate=np.where((nmo_ge_10C>=4) & (Ta_diff<=35) & ~np.isfinite(thermal_climate),np.float32(7),np.float32(thermal_climate))  #KLG
        # Continental Temperate  #KLG
        thermal_climate=np.where((nmo_ge_10C>=4) & (Ta_diff>35) & ~np.isfinite(thermal_climate),np.float32(8),np.float32(thermal_climate))  #KLG
        
        # Boreal  #KLG
        # Oceanic Boreal  #KLG
        thermal_climate=np.where((nmo_ge_10C>=1) & (Ta_diff<=20) & ~np.isfinite(thermal_climate),np.float32(9),np.float32(thermal_climate))  #KLG
        # Sub-Continental Boreal  #KLG
        thermal_climate=np.where((nmo_ge_10C>=1) & (Ta_diff<=35) & ~np.isfinite(thermal_climate),np.float32(10),np.float32(thermal_climate))  #KLG
        # Continental Boreal  #KLG
        thermal_climate=np.where((nmo_ge_10C>=1) & (Ta_diff>35) & ~np.isfinite(thermal_climate),np.float32(11),np.float32(thermal_climate))  #KLG
        
        # Arctic  #KLG
        thermal_climate=np.where(~np.isfinite(thermal_climate),np.float32(12),np.float32(thermal_climate))  #KLG

        # thermal_climate = np.zeros((self.im_height, self.im_width), dtype= np.int8)

        # for i_row in range(self.im_height):
        #     for i_col in range(self.im_width):

        #         if self.set_mask:
        #             if self.im_mask[i_row, i_col] == self.nodata_val:
        #                 continue
                
        #         # converting daily to monthly
        #         obj_utilities = UtilitiesCalc.UtilitiesCalc()
        #         meanT_monthly_sealevel = obj_utilities.averageDailyToMonthly(self.meanT_daily_sealevel[i_row,i_col,:])
        #         meanT_monthly = obj_utilities.averageDailyToMonthly(self.meanT_daily[i_row,i_col,:])
        #         P_by_PET_monthly = obj_utilities.averageDailyToMonthly(self.P_by_PET_daily[i_row,i_col,:])

        #         if self.set_mask:
        #             if self.im_mask[i_row, i_col] == self.nodata_val:
        #                 continue
                    
        #         # Seasonal parameters            
        #         summer_PET0 = np.sum(P_by_PET_monthly[3:9])
        #         winter_PET0 = np.sum([P_by_PET_monthly[9::], P_by_PET_monthly[0:3]])
        #         Ta_diff = np.max(meanT_monthly_sealevel) - \
        #             np.min(meanT_monthly_sealevel)
                
        #         # Tropics
        #         if np.min(meanT_monthly_sealevel) >= 18. and Ta_diff < 15.:
        #             if np.mean(meanT_monthly) < 20.:
        #                 thermal_climate[i_row, i_col] = 2  # Tropical highland
        #             else:
        #                 thermal_climate[i_row, i_col] = 1  # Tropical lowland
                        
        #         # SubTropic
        #         elif np.min(meanT_monthly_sealevel) >= 5. and np.sum(meanT_monthly_sealevel >= 10) >= 8:
        #             if np.sum(self.totalPrec_daily[i_row,i_col,:]) < 250:
        #                 # 'Subtropics Low Rainfall
        #                 thermal_climate[i_row,i_col] = 3
        #             elif self.latitude[i_row,i_col]>=0: 
        #                 if summer_PET0 >= winter_PET0:
        #                     # Subtropics Summer Rainfall
        #                     thermal_climate[i_row,i_col] = 4
        #                 else:
        #                     # Subtropics Winter Rainfall
        #                     thermal_climate[i_row,i_col] = 5
        #             else:
        #                 if summer_PET0 >= winter_PET0:
        #                     # Subtropics Winter Rainfall
        #                     thermal_climate[i_row,i_col] = 5                     
        #                 else:
        #                     # Subtropics Summer Rainfall
        #                     thermal_climate[i_row,i_col] = 4

                        
        #         # Temperate
        #         elif np.sum(meanT_monthly_sealevel >= 10) >= 4:
        #             if Ta_diff <= 20:
        #                 # Oceanic Temperate
        #                 thermal_climate[i_row, i_col] = 6
        #             elif Ta_diff <= 35:
        #                 # Sub-Continental Temperate
        #                 thermal_climate[i_row, i_col] = 7
        #             else:
        #                 # Continental Temperate
        #                 thermal_climate[i_row, i_col] = 8

        #         elif np.sum(meanT_monthly_sealevel >= 10) >= 1:
        #             # Boreal
        #             if Ta_diff <= 20:
        #                 # Oceanic Boreal
        #                 thermal_climate[i_row, i_col] = 9
        #             elif Ta_diff <= 35:
        #                 # Sub-Continental Boreal
        #                 thermal_climate[i_row, i_col] = 10
        #             else:
        #                 # Continental Boreal
        #                 thermal_climate[i_row, i_col] = 11
        #         else:
        #             # Arctic
        #             thermal_climate[i_row, i_col] = 12
        print('setting mask')                    
        if self.set_mask:
            if self.parallel:
                print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask
            thermal_climate=np.where(mask, thermal_climate.astype('float32'), np.float32(np.nan))  #KLG
            return thermal_climate
        else:
            return thermal_climate.astype('float32')#.compute()  #KLG

    

    def getThermalZone(self):
        """The thermal zone is classified based on actual temperature which reflects 
        on the temperature regimes of major thermal climates

        Returns:
            2D NumPy: Thermal Zones classification
        """        
        # if self.chunk3D:
        # else:
        thermal_zone = np.empty((self.im_height,self.im_width),dtype='float32')  #KLG
        thermal_zone[:] = np.nan        

        # converting daily to monthly
        # obj_utilities = UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D)  #KLG
        # print('computing meanT_monthly')        
        # meanT_monthly = obj_utilities.averageDailyToMonthly(self.meanT_daily)
        meanT_monthly=self.meanT_monthly
        # print('computing meanT_monthly_sealevel')        
        # meanT_monthly_sealevel = obj_utilities.averageDailyToMonthly(self.meanT_daily_sealevel)
        meanT_monthly_sealevel=self.meanT_monthly_sealevel

        # things we need to determine the classes
        # compute them here for readability below
        min_sealev_meanT=meanT_monthly_sealevel.min(axis=2)
        range_meanT=meanT_monthly.max(axis=2) - meanT_monthly.min(axis=2)
        # Ta_diff=meanT_monthly_sealevel.max(axis=2) - meanT_monthly_sealevel.min(axis=2)  #KLG
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
        thermal_zone=np.where((min_sealev_meanT>=18) & (range_meanT<15) & (meanT<=20) & ~np.isfinite(thermal_zone),2,thermal_zone)
        # Subtropics, cool
        thermal_zone=np.where((min_sealev_meanT>5) & (nmo_gt_10C_sealev>=8) & (nmo_lt_5C>=1) & (nmo_gt_10C>=4) & ~np.isfinite(thermal_zone),4,thermal_zone)
        # Subtropics, cold
        thermal_zone=np.where((min_sealev_meanT>5) & (nmo_gt_10C_sealev>=8) & (nmo_lt_5C>=1) & (nmo_gt_10C>=1) & ~np.isfinite(thermal_zone),5,thermal_zone)
        #Subtropics, very cold
        thermal_zone=np.where((min_sealev_meanT>5) & (nmo_gt_10C_sealev>=8) & (nmo_lt_10C==12) & ~np.isfinite(thermal_zone),6,thermal_zone)
        # Subtropics, warm/mod. cool
        thermal_zone=np.where((min_sealev_meanT>5) & (nmo_gt_10C_sealev>=8) & ~np.isfinite(thermal_zone),3,thermal_zone)        
        # Temperate, cool
        thermal_zone=np.where((nmo_ge_10C_sealev>=4) & (nmo_lt_5C>=1) & (nmo_gt_10C>=4) & ~np.isfinite(thermal_zone),7,thermal_zone)
        # Temperate, cold
        thermal_zone=np.where((nmo_ge_10C_sealev>=4) & (nmo_lt_5C>=1) & (nmo_gt_10C>=1) & ~np.isfinite(thermal_zone),8,thermal_zone)
        # Temperate, very cold
        thermal_zone=np.where((nmo_ge_10C_sealev>=4) & (nmo_lt_10C==12) & ~np.isfinite(thermal_zone),9,thermal_zone)
        # Boreal, cold
        thermal_zone=np.where((nmo_ge_10C_sealev>=1) & (nmo_lt_5C>=1) & (nmo_gt_10C>=1) & ~np.isfinite(thermal_zone),10,thermal_zone)
        # Boreal, very cold
        thermal_zone=np.where((nmo_ge_10C_sealev>=1) & (nmo_lt_10C==12) & ~np.isfinite(thermal_zone),11,thermal_zone)
        # Arctic
        thermal_zone=np.where(~np.isfinite(thermal_zone),12,thermal_zone)

        # abandon loops for vectorization = much faster compute    
        # for i_row in range(self.im_height):
        #     for i_col in range(self.im_width):
                
        #         obj_utilities = UtilitiesCalc.UtilitiesCalc()

        #         meanT_monthly = obj_utilities.averageDailyToMonthly(self.meanT_daily[i_row, i_col, :])
        #         meanT_monthly_sealevel =  obj_utilities.averageDailyToMonthly(self.meanT_daily_sealevel[i_row, i_col, :])
    
        #         if self.set_mask:
        #             if self.im_mask[i_row, i_col] == self.nodata_val:
        #                 continue
    
        #         if np.min(meanT_monthly_sealevel) >= 18 and np.max(meanT_monthly)-np.min(meanT_monthly) < 15:
        #             if np.mean(meanT_monthly) > 20:
        #                 thermal_zone[i_row,i_col] = 1 # Tropics Warm
        #             else:
        #                 thermal_zone[i_row,i_col] = 2 # Tropics cool/cold/very cold
                
        #         elif np.min(meanT_monthly_sealevel) > 5 and np.sum(meanT_monthly_sealevel > 10) >= 8:
        #             if np.sum(meanT_monthly<5) >= 1 and np.sum(meanT_monthly>10) >= 4:
        #                 thermal_zone[i_row,i_col] =  4 # Subtropics, cool
        #             elif np.sum(meanT_monthly<5) >= 1 and np.sum(meanT_monthly>10) >= 1:
        #                 thermal_zone[i_row,i_col] =  5 # Subtropics, cold
        #             elif np.sum(meanT_monthly<10) == 12:
        #                 thermal_zone[i_row,i_col] =  6 # Subtropics, very cold
        #             else:
        #                 thermal_zone[i_row,i_col] =  3 # Subtropics, warm/mod. cool
    
        #         elif np.sum(meanT_monthly_sealevel >= 10) >= 4:
        #             if np.sum(meanT_monthly<5) >= 1 and np.sum(meanT_monthly>10) >= 4:
        #                 thermal_zone[i_row,i_col] =  7 # Temperate, cool
        #             elif np.sum(meanT_monthly<5) >= 1 and np.sum(meanT_monthly>10) >= 1:
        #                 thermal_zone[i_row,i_col] =  8 # Temperate, cold
        #             elif np.sum(meanT_monthly<10) == 12:
        #                 thermal_zone[i_row,i_col] =  9 # Temperate, very cold
    
        #         elif np.sum(meanT_monthly_sealevel >= 10) >= 1:
        #             if np.sum(meanT_monthly<5) >= 1 and np.sum(meanT_monthly>10) >= 1:
        #                 thermal_zone[i_row,i_col] = 10 # Boreal, cold
        #             elif np.sum(meanT_monthly<10) == 12:
        #                 thermal_zone[i_row,i_col] = 11 # Boreal, very cold
        #         else:
        #                 thermal_zone[i_row,i_col] = 12 # Arctic
    
        if self.set_mask:
            if self.parallel:
                print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask            
            return np.where(mask, thermal_zone.astype('float32'), np.nan)  #KLG
        else:
            return thermal_zone.astype('float32')  #KLG

    def getThermalLGP0(self):
        """Calculate Thermal Length of Growing Period (LGPt) with 
        temperature threshold of 0 degree Celcius

        Returns:
            2D numpy: The accumulated number of days with daily mean 
                      temperature is above 0 degree Celcius
        """        
        # Adding interpolation to the dataset
        # interp_daily_temp = np.zeros((self.im_height, self.im_width, 365))

        lgpt0 = np.sum(self.meanT_daily>=0, axis=2)
        if self.set_mask:
            lgpt0 = np.where(self.im_mask,np.float32(lgpt0),np.float32(np.nan))
        
        if self.parallel:
            print('in ClimateRegime, computing lgpt0 in parallel')
            lgpt0=lgpt0.compute()

        self.lgpt0=lgpt0.copy()
        return lgpt0.astype('float32')


    def getThermalLGP5(self):
        """Calculate Thermal Length of Growing Period (LGPt) with 
        temperature threshold of 5 degree Celcius

        Returns:
            2D numpy: The accumulated number of days with daily mean 
                      temperature is above 5 degree Celcius
        """          
        lgpt5 = np.sum(self.meanT_daily>=5, axis=2)
        if self.set_mask:
            lgpt5 = np.where(self.im_mask,np.float32(lgpt5),np.float32(np.nan))
        
        if self.parallel:
            print('in ClimateRegime, computing lgpt5 in parallel')
            lgpt5=lgpt5.compute()

        self.lgpt5 = lgpt5.copy()
        return lgpt5.astype('float32')

    def getThermalLGP10(self):
        """Calculate Thermal Length of Growing Period (LGPt) with
        temperature threshold of 10 degree Celcius

        Returns:
            2D numpy: The accumulated number of days with daily mean
                      temperature is above 10 degree Celcius
        """

        lgpt10 = np.sum(self.meanT_daily >= 10, axis=2)
        if self.set_mask:
            lgpt10 = np.where(self.im_mask, np.float32(lgpt10), np.float32(np.nan))
        
        if self.parallel:
            print('in ClimateRegime, computing lgpt10 in parallel')
            lgpt10=lgpt10.compute()

        self.lgpt10 = lgpt10.copy()
        return lgpt10.astype('float32')

    def getTemperatureSum0(self):
        """Calculate temperature summation at temperature threshold 
        of 0 degree Celcius

        Returns:
            2D numpy: Accumulative daily average temperature (Ta) for days
                      when Ta is above the thresholds of 0 degree Celcius
        """
        tempT = self.meanT_daily.copy()
        tempT=np.where(tempT<0,0,tempT)
        tsum0 = np.round(np.sum(tempT, axis=2), decimals = 0) 

        # masking
        if self.set_mask:
            tsum0 = np.where(self.im_mask, np.float32(tsum0), np.float32(np.nan))
        
        if self.parallel:
            print('in ClimateRegime, computing tsum0 in parallel')
            tsum0=tsum0.compute()
        return tsum0

    def getTemperatureSum5(self):
        """Calculate temperature summation at temperature threshold 
        of 5 degree Celcius

        Returns:
            2D numpy: Accumulative daily average temperature (Ta) for days
                      when Ta is above the thresholds of 5 degree Celcius
        """
        tempT = self.meanT_daily.copy()
        tempT=np.where(tempT<5,0,tempT)
        tsum5 = np.round(np.sum(tempT, axis=2), decimals = 0) 

        # masking
        if self.set_mask: 
            tsum5 = np.where(self.im_mask, np.float32(tsum5), np.float32(np.nan))

        if self.parallel:
            print('in ClimateRegime, computing tsum5 in parallel')
            tsum5=tsum5.compute()
        return tsum5
        

    def getTemperatureSum10(self):
        """Calculate temperature summation at temperature threshold 
        of 10 degree Celcius

        Returns:
            2D numpy: Accumulative daily average temperature (Ta) for days
                      when Ta is above the thresholds of 10 degree Celcius
        """
        tempT = self.meanT_daily.copy()
        tempT=np.where(tempT<10,0,tempT)
        tsum10 = np.round(np.sum(tempT, axis=2), decimals = 0) 

        # masking
        if self.set_mask: 
            tsum10 = np.where(self.im_mask, np.float32(tsum10), np.float32(np.nan))

        if self.parallel:
            print('in ClimateRegime, computing tsum10 in parallel')
            tsum10=tsum10.compute()            
        return tsum10

    def getTemperatureProfile(self):
        """Classification of temperature ranges for temperature profile

        Returns:
            2D NumPy: 18 2D arrays [A1-A9, B1-B9] correspond to each Temperature Profile class [days]
        """    
        # list of variable names to compute and output  #KLG
        # var_names = ['A1','A2','A3','A4','A5','A6','A7','A8','A9', \
        #             'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']  #KLG
        # var_dict=OrderedDict.fromkeys(var_names)  #KLG

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
        print('1:',psutil.virtual_memory().free/1E9)
        # smoothed daily temperature
        # create a mask of all 1's if the user doesn't provide a mask
        if self.set_mask:
            # if self.parallel:
            #     print('in ClimateRegime, computing mask in parallel')
            #     mask=self.im_mask.compute()
            # else:
            mask=self.im_mask 
        else:
            mask=np.ones((self.im_height,self.im_width),dtype='int')

        # print(psutil.virtual_memory().free/1E9)
        obj_utilities = UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D)  #KLG
        meanT_first=obj_utilities.smoothDailyTemp(self.doy_start,self.doy_end, mask, self.meanT_daily)


        # meanT_first=self.interp_daily_temp  #KLG
        print('computing meanT_diff')
        # meanT_diff=np.diff(self.interp_daily_temp,n=1,axis=2,append=self.interp_daily_temp[:,:,0:1])   #KLG
        # meanT_diff=np.diff(meanT_first,n=1,axis=2,append=meanT_first[:,:,0:1]).astype('float32')   #KLG
        meanT_diff=da.diff(meanT_first,n=1,axis=2,append=meanT_first[:,:,0:1]).astype('float32').rechunk(self.chunk3D)   #KLG


        # var_dict['A9'] = np.sum( np.logical_and(meanT_diff>0, meanT_first<-5), axis=2 )  #KLG
        # var_dict['A8'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=-5, meanT_first<0)), axis=2 )  #KLG
        # var_dict['A7'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=0, meanT_first<5)), axis=2 )  #KLG
        # var_dict['A6'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=5, meanT_first<10)), axis=2 )  #KLG
        # var_dict['A5'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=10, meanT_first<15)), axis=2 )  #KLG
        # var_dict['A4'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=15, meanT_first<20)), axis=2 )  #KLG
        # var_dict['A3'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=20, meanT_first<25)), axis=2 )  #KLG
        # var_dict['A2'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=25, meanT_first<30)), axis=2 )  #KLG
        # var_dict['A1'] = np.sum( np.logical_and(meanT_diff>0, meanT_first>=30), axis=2 )  #KLG

        # var_dict['B9'] = np.sum( np.logical_and(meanT_diff<0, meanT_first<-5), axis=2 )  #KLG
        # var_dict['B8'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=-5, meanT_first<0)), axis=2 )  #KLG
        # var_dict['B7'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=0, meanT_first<5)), axis=2 )  #KLG
        # var_dict['B6'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=5, meanT_first<10)), axis=2 )  #KLG
        # var_dict['B5'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=10, meanT_first<15)), axis=2 )  #KLG
        # var_dict['B4'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=15, meanT_first<20)), axis=2 )  #KLG
        # var_dict['B3'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=20, meanT_first<25)), axis=2 )  #KLG
        # var_dict['B2'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=25, meanT_first<30)), axis=2 )   #KLG       
        # var_dict['B1'] = np.sum( np.logical_and(meanT_diff<0, meanT_first>=30), axis=2 )  #KLG

        # delay the input data so it's copied once instead of at each call of the function
        print('delaying data inputs')
        meanT_diff=meanT_diff.to_delayed()
        # meanT_diff=dask.delayed(meanT_diff)
        # meanT_first=dask.delayed(self.interp_daily_temp)
        # meanT_first=dask.delayed(meanT_first)
        meanT_first=meanT_first.to_delayed()
        mask=mask.to_delayed()

        # put the computations inside a delayed function
        @dask.delayed
        def sum_ndays_per_tprof_class(diff,meanT,tendency,lim_lo,lim_hi,mask):
            if tendency=='warming':
                tclass_ndays = np.sum( (diff>0)&(meanT>=lim_lo)&(meanT<lim_hi), axis=2 ) 
            if tendency=='cooling':
                tclass_ndays = np.sum( (diff<0)&(meanT>=lim_lo)&(meanT<lim_hi), axis=2 )

            # apply mask
            tclass_ndays=np.where(mask, tclass_ndays.astype('float32'), np.float32(np.nan))
            return tclass_ndays

        # in a regular non-delayed loop, call delayed function and compile list of future compute tasks
        print('building task list')
        task_list=[]                        
        for class_inputs in tclass_info.values():
            for d,t,m in zip(meanT_diff.ravel(),meanT_first.ravel(),mask.ravel()):
                task=sum_ndays_per_tprof_class(d,t,class_inputs['tendency'],class_inputs['lim_lo'],class_inputs['lim_hi'],m)
                task_list.append(task)

        
        # compute tasks in parallel
        # this returns a list of arrays in the same order as tclass_info
        print('in ClimateRegime, computing temperature profiles in parallel')
        data_out=dask.compute(*task_list)

        # concatenate chunks
        tprofiles=[]
        for i,key in enumerate(tclass_info.keys()):
            # print(key)
            tprofiles.append(np.concatenate(data_out[i*self.nchunks:i*self.nchunks+self.nchunks],axis=1))

        # apply the mask  #KLG
        # if self.set_mask:  #KLG
        #     for var in var_names:  #KLG
        #         var_dict[var]=np.ma.masked_where(self.im_mask == 0, var_dict[var])  #KLG

        # # assemble the list of return variables  #KLG
        # # the order of the variables in the returned list is the same as in var_names  #KLG
        # data_out=[]  #KLG
        # for var in var_names:  #KLG
        #     data_out.append(var_dict[var].astype('float32'))  #KLG
        
        # # return data_out   
        return tprofiles   
        # return meanT_first           


    def getLGP(self, Sa=100., D=1.):
        """Calculate length of growing period (LGP)

        Args:
            Sa (float, optional): Available soil moisture holding capacity [mm/m]. Defaults to 100..
            D (float, optional): Rooting depth. Defaults to 1..

        Returns:
           2D NumPy: Length of Growing Period
        """   
        #============================
        # 1) prep inputs chunked like (ny,nx,365)
        # 2) call loop function on chunks, return lgp_tot
        # 3) concat lgp_tot



        # #============================
        # ### CALCULATE ISTART0,ISTART1 ###
        
        # # create delayed chunks
        # lgpt5_delayed=da.from_array(self.lgpt5,chunks=self.chunk2D).to_delayed().ravel()
        # meanT_delayed=self.meanT_daily.to_delayed().ravel()
        # # create list of delayed compute tasks on array chunks and then compute
        # task_list=[dask.delayed(LGPCalc.rainPeak)(meanT_chunk,lgp_chunk) for meanT_chunk,lgp_chunk in zip(meanT_delayed,lgpt5_delayed)]
        # results_list=dask.compute(*task_list)
        # # separate results into lists of chunks
        # istart0_chunks=[]
        # istart1_chunks=[]
        # for c in range(self.nchunks):
        #     istart0_chunks.append(results_list[c][0])
        #     istart1_chunks.append(results_list[c][1])
        # # concatenate to numpy array
        # istart0_delayed=dask.delayed(np.concatenate(istart0_chunks,axis=1))
        # istart1_delayed=dask.delayed(np.concatenate(istart1_chunks,axis=1))
        # # clean up
        # del meanT_delayed,lgpt5_delayed,results_list#,istart0_chunks,istart1_chunks    

        # #============================
        # ### SET UP INPUTS TO THE DAY LOOP ###
        # ### ALL INPUTS ARE DELAYED OBJECTS OF SHAPE (NY,NX) OR CONSTANTS ###
        # kc_list = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
        # Txsnm = 0.  # Txsnm - snow melt temperature threshold
        # Fsnm = 5.5  # Fsnm - snow melting coefficient
        # # Sb_old = np.zeros((self.im_height,self.im_width),dtype='float32')  #KLG
        # # Wb_old = np.zeros((self.im_height,self.im_width),dtype='float32')  #KLG    
        # # lgpt5_delayed=dask.delayed(self.lgpt5)   
        # Sb_old = da.zeros((self.im_height,self.im_width),chunks=self.chunk2D,dtype='float32')  #KLG
        # Wb_old = da.zeros((self.im_height,self.im_width),chunks=self.chunk2D,dtype='float32')  #KLG 
        # lgpt5_delayed=da.from_array(self.lgpt5,chunks=self.chunk2D).blocks.ravel()  
        # timechunks=(1,-1,self.chunk3D[1])
        # Tx365_timechunked = self.maxT_daily.transpose((2,0,1)).rechunk(chunks=timechunks).blocks.ravel()
        # Ta365_timechunked = self.meanT_daily.transpose((2,0,1)).rechunk(chunks=timechunks).blocks.ravel()
        # Pcp365_timechunked = self.totalPrec_daily.transpose((2,0,1)).rechunk(chunks=timechunks).blocks.ravel()
        # # timechunks=(1,-1,-1) # move time to the first dimension
        # # Tx365_timechunked = self.maxT_daily.transpose((2,0,1)).rechunk(chunks=timechunks).blocks.ravel()
        # # Ta365_timechunked = self.meanT_daily.transpose((2,0,1)).rechunk(chunks=timechunks).blocks.ravel()
        # # Pcp365_timechunked = self.totalPrec_daily.transpose((2,0,1)).rechunk(chunks=timechunks).blocks.ravel()
        # # arrays to hold the outputs
        # self.Etm365 = np.empty(self.meanT_daily.shape,dtype='float32')  #KLG
        # self.Eta365 = np.empty(self.meanT_daily.shape,dtype='float32')  #KLG        
        # self.Etm365[:],self.Eta365[:]=np.nan,np.nan  
        # # mask       
        # if self.set_mask:
        #     if self.parallel:
        #         print('in ClimateRegime, computing mask in parallel')
        #         # mask_delayed=dask.delayed(self.im_mask.compute())
        #         mask_delayed=self.im_mask.compute()
        #     else:
        #         mask_delayed=dask.delayed(self.im_mask)
        # else:
        #     mask_delayed=dask.delayed(np.ones((self.im_height,self.im_width),dtype='int'))

        # pet_timechunked=da.from_array(self.pet_daily.transpose((2,0,1))).rechunk(chunks=timechunks).blocks.ravel()
        # task_list = [dask.delayed(LGPCalc.psh)(np.zeros((self.im_height,self.im_width),dtype='float32'),pet) for pet in pet_timechunked]   #KLG            
        # p_chunks=dask.compute(*task_list)
        # p_delayed=[dask.delayed(arr) for arr in p_chunks]
        # del p_chunks
        # # p = LGPCalc.psh(np.zeros(self.pet_daily.shape,dtype='float32'), self.pet_daily)   #KLG

        # start=timer()
        # lgpt5=da.from_array(self.lgpt5,chunks=self.chunk2D)
        # istart0,istart1=LGPCalc.rainPeak(self.meanT_daily,lgpt5)
        # # task_time=timer()-start
        # # print('time spent on istart0,istart1',task_time)

        # # start=timer()
        # # pet_daily=da.from_array(self.pet_daily,chunks=self.chunk3D)
        # pet_daily=self.pet_daily
        # # task_time=timer()-start
        # # print('time spent on converting pet_daily',task_time)
        # # start=timer()
        # p = LGPCalc.psh(da.zeros(pet_daily.shape,chunks=self.chunk3D,dtype='float32'), pet_daily)
        # # task_time=timer()-start
        # # print('time spent on p',task_time)

        # # start=timer()
        # ndays=self.maxT_daily.shape[2]
        # lgpt5_3D=da.tile(self.lgpt5[:,:,np.newaxis],(1,1,ndays)).rechunk(chunks=self.chunk3D) 
        # mask_3D=da.tile(self.im_mask[:,:,np.newaxis],(1,1,ndays)).rechunk(chunks=self.chunk3D)
        # # task_time=timer()-start
        # # print('time spent on 3D vars',task_time)          
        # # start=timer()
        # Txsnm = 0. 
        # eta_class=LGPCalc.Eta_class(mask_3D,lgpt5_3D,self.meanT_daily,self.maxT_daily,Txsnm)
        # del lgpt5_3D,mask_3D
        # # task_time=timer()-start
        # # print('time spent on eta_class',task_time)       

        # islgp=da.where(self.meanT_daily>=5,np.int8(1),np.int8(0))  #KLG

        # # constants
        # Fsnm = 5.5
        # kc_list = np.array([0.0, 0.1, 0.2, 0.5, 1.0],dtype='float32')   

        # smallchunk3D=(-1,1,-1)#self.chunk3D#(-1,400,-1)
        # smallchunk2D=(-1,1)#self.chunk2D#(-1,400)
        # # 2D chunked inputs
        # mask_cl=self.im_mask.rechunk(chunks=smallchunk2D).to_delayed().ravel()#.blocks.ravel()
        # lgpt5_cl = lgpt5.rechunk(chunks=smallchunk2D).to_delayed().ravel()#.blocks.ravel()
        # istart0_cl = istart0.rechunk(chunks=smallchunk2D).to_delayed().ravel()#.blocks.ravel()
        # istart1_cl = istart1.rechunk(chunks=smallchunk2D).to_delayed().ravel()#.blocks.ravel()
        # # Sb_old_cl = da.zeros((self.im_height,self.im_width),chunks=self.chunk2D,dtype='float32')#.to_delayed().ravel()#.blocks.ravel()
        # # Wb_old_cl = da.zeros((self.im_height,self.im_width),chunks=self.chunk2D,dtype='float32')#.to_delayed().ravel()#.blocks.ravel()         
        # Sb_old_cl = da.zeros((self.im_height,self.im_width),chunks=smallchunk2D,dtype='float32').to_delayed().ravel()#.blocks.ravel()
        # Wb_old_cl = da.zeros((self.im_height,self.im_width),chunks=smallchunk2D,dtype='float32').to_delayed().ravel()#.blocks.ravel() 

        # # 3D chunked inputs
        # Pet365_cl = pet_daily.rechunk(chunks=smallchunk3D).to_delayed().ravel()#.blocks.ravel()
        # p_cl = p.rechunk(chunks=smallchunk3D).to_delayed().ravel()#.blocks.ravel()
        # eta_class_cl = eta_class.rechunk(chunks=smallchunk3D).to_delayed().ravel()#.blocks.ravel()
        # Tx365_cl = self.maxT_daily.rechunk(chunks=smallchunk3D).to_delayed().ravel()#.blocks.ravel()
        # # Ta365_cl = self.meanT_daily.rechunk(chunks=smallchunk3D).to_delayed().ravel()#.blocks.ravel()
        # Pcp365_cl = self.totalPrec_daily.rechunk(chunks=smallchunk3D).to_delayed().ravel()#.blocks.ravel()
        # islgp_cl = islgp.rechunk(chunks=smallchunk3D).to_delayed().ravel()

        # # zip all the chunked variables together
        # # zipvars=zip(mask_cl, Tx365_cl, Ta365_cl, Pcp365_cl, Pet365_cl, Wb_old_cl, Sb_old_cl, istart0_cl, istart1_cl, p_cl, lgpt5_cl, eta_class_cl)
        # zipvars=zip(mask_cl, Tx365_cl, islgp_cl, Pcp365_cl, Pet365_cl, Wb_old_cl, Sb_old_cl, istart0_cl, istart1_cl, p_cl, lgpt5_cl, eta_class_cl)
        # # zipvars=zip(mask_cl[0:2], Tx365_cl[0:2], Ta365_cl[0:2], Pcp365_cl[0:2], Pet365_cl[0:2], Wb_old_cl[0:2], Sb_old_cl[0:2], istart0_cl[0:2], istart1_cl[0:2], p_cl[0:2], lgpt5_cl[0:2], eta_class_cl[0:2])
        # # print(mask_cl[0])
        # # print(len(mask_cl), len(Tx365_cl), len(Ta365_cl), len(Pcp365_cl), len(Pet365_cl), len(Wb_old_cl), len(Sb_old_cl), len(istart0_cl), len(istart1_cl), len(p_cl), len(lgpt5_cl), len(eta_class_cl))
        # # print(len(mask_cl), len(Tx365_cl), len(islgp_cl), len(Pcp365_cl), len(Pet365_cl), len(Wb_old_cl), len(Sb_old_cl), len(istart0_cl), len(istart1_cl), len(p_cl), len(lgpt5_cl), len(eta_class_cl))
        # task_time=timer()-start
        # print('time spent on prepping vars',task_time) 

        # start=timer()
        # # create task_list, one chunk of each arr per task
        # # task_list=[dask.delayed(LGPCalc.EtaCalc)(mask,Tx365,Ta365,Pcp365,Txsnm,Fsnm,pet365,Wb_old,Sb_old,istart0,istart1,Sa,D,p,kc_list,lgpt5,eclass,self.doy_start,self.doy_end) for mask,Tx365,Ta365,Pcp365,pet365,Wb_old,Sb_old,istart0,istart1,p,lgpt5,eta_class in zipvars]
        # task_list=[dask.delayed(LGPCalc.EtaCalc, traverse=False)(mask,Tx365,blgp,Pcp365,Txsnm,Fsnm,pet365,Wb_old,Sb_old,s0,s1,Sa,D,p_c,kc_list,lgp,eclass,self.doy_start,self.doy_end) for mask,Tx365,blgp,Pcp365,pet365,Wb_old,Sb_old,s0,s1,p_c,lgp,eclass in zipvars]
        # print('NUMBER OF TASKS',len(task_list))
        # task_time=timer()-start
        # print('time spent building task list',task_time) 
        # print('free RAM',psutil.virtual_memory().free/1E9)
        
        # start=timer()
        # results=dask.compute(*task_list, traverse=False)
        # task_time=timer()-start
        # print('time spent in compute',task_time)    
        # # print(len(results))
        # # print(len(results))
        # # print(results[0].shape)
        # # test=list(results[0])
        # # print(test)

        start=timer()
        bigchunk2D=(-1,72)#(-1,80)#(-1,90)#(-1,132)#
        bigchunk3D=(-1,72,-1)#(-1,80,-1)#(-1,90,-1)#(-1,132,-1)#
        nchunks=60#54#48#33#54
        Txsnm = 0. 
        #'build task graph for istart0,istart1,p'
        lgpt5=da.from_array(self.lgpt5,chunks=bigchunk2D)
        istart0,istart1=LGPCalc.rainPeak(self.meanT_daily.rechunk(chunks=bigchunk3D),lgpt5)
        ng=da.zeros(self.pet_daily.shape,chunks=bigchunk3D,dtype='float32')
        pet=da.from_array(self.pet_daily,chunks=bigchunk3D)        
        #'compute eta_class'
        lgpt5_3D=da.broadcast_to(self.lgpt5[:,:,np.newaxis].astype('float16'),(self.im_height,self.im_width,self.doy_end)).rechunk(chunks=bigchunk3D).to_delayed().ravel()
        mask_3D=da.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,self.doy_end)).rechunk(chunks=bigchunk3D).to_delayed().ravel()
        Tmean=self.meanT_daily.rechunk(chunks=bigchunk3D).astype('float16').to_delayed().ravel()
        Tmax=self.maxT_daily.rechunk(chunks=bigchunk3D).astype('float16').to_delayed().ravel()
        zip1=zip(mask_3D,lgpt5_3D,Tmean,Tmax)
        task_list=[dask.delayed(LGPCalc.Eta_class)(m,l5,Tbar,Tmx,Txsnm) for m,l5,Tbar,Tmx in zip1]
        results=dask.compute(*task_list)
        eta_class=np.concatenate(results,axis=1)
        del lgpt5_3D,mask_3D,Tmean,Tmax,zip1,task_list,results
        #'build task graph for islgp'
        islgp=da.where(self.meanT_daily>=5,np.int8(1),np.int8(0)).rechunk(chunks=bigchunk3D)  #KLG
        Fsnm = 5.5
        kc_list = np.array([0.0, 0.1, 0.2, 0.5, 1.0],dtype='float32')

        # chunk all inputs to big chunks
        lgpt5_c=lgpt5
        mask_c=self.im_mask.rechunk(chunks=bigchunk2D)
        istart0_c = istart0
        istart1_c = istart1  
        Sb_old_c = da.zeros((self.im_height,self.im_width),chunks=bigchunk2D,dtype='float32')
        Wb_old_c = da.zeros((self.im_height,self.im_width),chunks=bigchunk2D,dtype='float32')
        Pet365_c = da.from_array(self.pet_daily,chunks=bigchunk3D)
        p_c = LGPCalc.psh(ng,pet)
        eta_class_c=da.from_array(eta_class,chunks=bigchunk3D)
        Tx365_c = self.maxT_daily.rechunk(chunks=bigchunk3D)
        Pcp365_c = self.totalPrec_daily.rechunk(chunks=bigchunk3D)
        islgp_c = islgp  
        task_time=timer()-start
        print('time spent on prepping vars',task_time) 

        start=timer()
        # this is not a normal way to compute with dask
        # our functions are so complicated that allowing dask to automate the parallel 
        # computation is much slower and/or crashes due to high memory use
        # here we loop thru chunks one at a time, compute the inputs ahead 
        # of time to reduce passing many task graphs, and call the EtaCalc 
        # func (which includes some parallelism) on each chunk, then concat the result chunks
        results=[]
        task_list=[]
        for i in range(nchunks):
            print('loop',i)
            mask_s=mask_c.blocks[0,i].compute()
            Tx365_s=Tx365_c.blocks[0,i,0].compute()
            islgp_s=islgp_c.blocks[0,i,0].compute()
            Pcp365_s=Pcp365_c.blocks[0,i,0].compute()
            Pet365_s=Pet365_c.blocks[0,i,0].compute()
            Wb_old_s=Wb_old_c.blocks[0,i].compute()
            Sb_old_s=Sb_old_c.blocks[0,i].compute()
            istart0_s=istart0_c.blocks[0,i].compute()
            istart1_s=istart1_c.blocks[0,i].compute()
            lgpt5_s=lgpt5_c.blocks[0,i].compute()
            # print('eta_class')
            eta_class_s=eta_class_c.blocks[0,i].compute()
            # print('p')
            p_s=p_c.blocks[0,i,0].compute()
            # print('lgp_tot')
            results.append(LGPCalc.EtaCalc(mask_s,Tx365_s,islgp_s,Pcp365_s,Txsnm,Fsnm,Pet365_s,Wb_old_s,Sb_old_s,istart0_s,istart1_s,Sa,D,p_s,kc_list,lgpt5_s,eta_class_s,self.doy_start,self.doy_end))
        task_time=timer()-start
        print('time spent in compute',task_time)  

        start=timer()
        lgp_tot=np.concatenate(results,axis=1)
        task_time=timer()-start
        print('time spent on lgp_tot concat',task_time)   

        # start=timer()
        # lgp_tot=da.map_blocks(LGPCalc.EtaCalc,mask_cl, Tx365_cl, Ta365_cl, Pcp365_cl,Txsnm,Fsnm,Pet365_cl, Wb_old_cl, Sb_old_cl, istart0_cl, istart1_cl,Sa,D,p_cl,kc_list,lgpt5_cl, eta_class_cl,self.doy_start,self.doy_end,dtype='float32',meta=np.array((self.im_height,self.im_width), dtype='float32')).compute()         
        # task_time=timer()-start
        # print('time spent in compute',task_time)  

        # #### NEXT STEP #################
        # ###### MANUALLY CHUNK PET_DAILY, P, ISTART0, ISTART1 TO DELAYED CHUNKS
        # for idoy in range(self.doy_start-1, self.doy_end):  #KLG
        #     print('in ClimateRegime, computing ET components for day',idoy+1)
        # # for idoy in range(0,2):  #KLG
        #     # istart=self.nchunks*idoy
        #     # iend=istart+self.nchunks
        #     # Eta_new, Etm_new, Wb_new, Wx_new, Sb_new, kc_new = LGPCalc.EtaCalc(
        #     Eta_new, Etm_new, Wb_new, Sb_new = LGPCalc.EtaCalc(
        #                             mask_delayed,
        #                             Tx365_timechunked[idoy], 
        #                             Ta365_timechunked[idoy],
        #                             Pcp365_timechunked[idoy], 
        #                             Txsnm, 
        #                             Fsnm, 
        #                             pet_timechunked[idoy],
        #                             Wb_old, 
        #                             Sb_old, 
        #                             idoy, 
        #                             istart0_delayed, 
        #                             istart1_delayed,
        #                             Sa, 
        #                             D, 
        #                             p_delayed[idoy], 
        #                             kc_list, 
        #                             lgpt5_delayed)  #KLG            

        #     self.Eta365[:,:,idoy]=Eta_new  #KLG
        #     self.Etm365[:,:,idoy]=Etm_new  #KLG
        #     # self.Wb365[:,:,doy]=Wb_new  #KLG
        #     # self.Wx365[:,:,doy]=Wx_new  #KLG
        #     # self.Sb365[:,:,doy]=Sb_new  #KLG
        #     # self.kc365[:,:,doy]=kc_new  #KLG

        #     Wb_old=Wb_new.copy()  #KLG
        #     Sb_old=Sb_new.copy()  #KLG

        # self.Eta365=np.where(self.Eta365<0,0,self.Eta365)  #KLG

        # Etm365X = np.append(self.Etm365, self.Etm365[:,:,0:30],axis=2)  #KLG
        # Eta365X = np.append(self.Eta365, self.Eta365[:,:,0:30],axis=2)  #KLG

        # # eliminate call to LGPCalc.islgpt  #KLG
        # islgp=np.where(self.meanT_daily>=5,np.int8(1),np.int8(0))  #KLG
        # if self.parallel:
        #     print('in ClimateRegime, computing islgp in parallel')
        #     islgp=islgp.compute()

        # xx = LGPCalc.val10day(Eta365X)  #KLG
        # yy = LGPCalc.val10day(Etm365X)  #KLG
    
        # with np.errstate(divide='ignore', invalid='ignore'):  #KLG
        #     lgp_whole = xx[:,:,:self.doy_end]/yy[:,:,:self.doy_end]  #KLG

        # lgp_tot=np.where((islgp==1)&(lgp_whole>=0.4),1,0).sum(axis=2)  #KLG
        
        if self.set_mask:
            if self.parallel:
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask
            return np.where(mask, lgp_tot, np.nan).astype('float32')  #KLG
        else:
            return lgp_tot.astype('float32')  #KLG
        # return results
  
    def getLGPClassified(self, lgp): # Original PyAEZ source code
        """This function calculates the classification of moisture regime using LGP.

        Args:
            lgp (2D NumPy): Length of Growing Period

        Returns:
            2D NumPy: Classified Length of Growing Period
        """        
        # 

        lgp_class = np.zeros(lgp.shape,dtype='float32')  #KLG

        lgp_class=np.where(lgp>=365,7,lgp_class) # Per-humid
        lgp_class=np.where((lgp>=270)&(lgp<365),6,lgp_class) # Humid
        lgp_class=np.where((lgp>=180)&(lgp<270),5,lgp_class) # Sub-humid
        lgp_class=np.where((lgp>=120)&(lgp<180),4,lgp_class) # Moist semi-arid
        lgp_class=np.where((lgp>=60)&(lgp<120),3,lgp_class) # Dry semi-arid
        lgp_class=np.where((lgp>0)&(lgp<60),2,lgp_class) # Arid
        lgp_class=np.where(lgp<=0,1,lgp_class) # Hyper-arid

        if self.set_mask:
            if self.parallel:
                print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask
            return np.where(mask, lgp_class, np.nan).astype('float32')
        else:
            return lgp_class
        
        
    def getLGPEquivalent(self): 
        """Calculate the Equivalent LGP 

        Returns:
            2D NumPy: LGP Equivalent 
        """        
        # if self.parallel:
        #     precipitation=self.totalPrec_daily.compute()
        # else:
        #     precipitation=self.totalPrec_daily

        # moisture_index = np.sum(precipitation,axis=2)/np.sum(self.pet_daily, axis=2)
        moisture_index = self.annual_accPrec/np.sum(self.pet_daily, axis=2)

        lgp_equv = 14.0 + 293.66*moisture_index - 61.25*moisture_index*moisture_index
        lgp_equv=np.where(moisture_index>2.4,366,lgp_equv)

        if self.set_mask:
            if self.parallel:
                print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask
            return np.where(mask, lgp_equv.astype('float32'), np.nan)  #KLG
        else:
            return lgp_equv

        '''
        Existing Issue: The moisture index calculation is technical aligned with FORTRAN routine, 
        results are still different from GAEZ; causing large discrepancy. 
        Overall, there are no changes with the calculation steps and logics.
        '''
      



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

        # the algorithm needs to calculate the annual mean temperature.
        tzonefallow = np.zeros((self.im_height, self.im_width), dtype= int)
        # if self.parallel:
        #     annual_Tmean = np.mean(self.meanT_daily, axis = 2).compute()
        # else:
        #     annual_Tmean = np.mean(self.meanT_daily, axis = 2)            
        annual_Tmean=self.annual_Tmean
        
        # obj_utilities = UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D)
        # max_meanTmonthly = obj_utilities.averageDailyToMonthly(self.meanT_daily).max(axis=2)  #KLG
        max_meanTmonthly=self.meanT_monthly.max(axis=2)

        # thermal zone class definitions for fallow requirement
        # for i_row in range(self.im_height):
        #     for i_col in range(self.im_width):

        #         if self.set_mask:
        #             if self.im_mask[i_row, i_col] == self.nodata_val:
        #                 continue
        #         # Checking tropics thermal zone
        #         if tzone[i_row, i_col] == 1 or tzone[i_row, i_col] == 2:
                    
        #             # Class 1: tropics, mean annual T > 25 deg C
        #             if annual_Tmean[i_row, i_col] > 25:
        #                 tzonefallow[i_row, i_col] = 1
                    
        #             # Class 2: tropics, mean annual T 20-25 deg C
        #             elif annual_Tmean[i_row, i_col] > 20:
        #                 tzonefallow[i_row, i_col] = 2
                    
        #             # Class 3: tropics, mean annual T 15-20 deg C
        #             elif annual_Tmean[i_row, i_col] > 15:
        #                 tzonefallow[i_row, i_col] = 3
                    
        #             # Class 4: tropics, mean annual T < 15 deg C
        #             else:
        #                 tzonefallow[i_row, i_col] = 4
                
        #         # Checking the non-tropical zones
        #         else:
        #             meanT_monthly = obj_utilities.averageDailyToMonthly(self.meanT_daily[i_row, i_col, :])
        #             # Class 5: mean T of the warmest month > 20 deg C
        #             if np.max(meanT_monthly) > 20:
        #                 tzonefallow[i_row, i_col] = 5
                        
        #             else:
        #                 tzonefallow[i_row, i_col] = 6

        # Checking tropics thermal zone
        # Class 1: tropics, mean annual T > 25 deg C
        tzonefallow=np.where(((tzone==1)|(tzone==2))&(annual_Tmean>25),1,tzonefallow)  #KLG
        # Class 2: tropics, mean annual T 20-25 deg C  #KLG
        tzonefallow=np.where(((tzone==1)|(tzone==2))&(annual_Tmean>20)&(tzonefallow==0),2,tzonefallow)  #KLG
        # Class 3: tropics, mean annual T 15-20 deg C  #KLG
        tzonefallow=np.where(((tzone==1)|(tzone==2))&(annual_Tmean>15)&(tzonefallow==0),3,tzonefallow)  #KLG
        # Class 4: tropics, mean annual T < 15 deg C  #KLG
        tzonefallow=np.where(((tzone==1)|(tzone==2))&(annual_Tmean<=15)&(tzonefallow==0),4,tzonefallow)  #KLG
        # Checking the non-tropical zones  #KLG
        # Class 5: mean T of the warmest month > 20 deg C  #KLG
        tzonefallow=np.where((tzone!=1)&(tzone!=2)&(max_meanTmonthly>20)&(tzonefallow==0),5,tzonefallow)  #KLG
        tzonefallow=np.where((tzone!=1)&(tzone!=2)&(max_meanTmonthly<=20)&(tzonefallow==0),6,tzonefallow) #KLG   

                            
        if self.set_mask:
            if self.parallel:
                print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask            
            return np.where(mask, tzonefallow.astype('float32'), np.nan) #KLG
        else:
            return tzonefallow.astype('float32') #KLG
    
  
    
   
    
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
        # fi = np.zeros((self.im_height, self.im_width), dtype=float)
        permafrost = np.zeros((self.im_height, self.im_width), dtype=int) 
        # ddt = np.zeros((self.im_height, self.im_width), dtype=float) # thawing index
        # ddf = np.zeros((self.im_height, self.im_width), dtype=float) # freezing index

        if self.parallel:
            print('in ClimateRegime, computing meanT_daily in parallel')
            meanT_gt_0 = self.meanT_daily.compute().copy()
            meanT_le_0 = meanT_gt_0            
        else:
            meanT_gt_0 = self.meanT_daily.copy()
            meanT_le_0 = self.meanT_daily.copy()
        
        meanT_gt_0=np.where(meanT_gt_0 <=0, np.float32(0), np.float32(meanT_gt_0)) # removing all negative temperatures for summation
        meanT_le_0=np.where(meanT_le_0 >0, np.float32(0), np.float32(meanT_le_0)) # removing all positive temperatures for summation #KLG
        ddt = np.sum(meanT_gt_0, axis = 2) # thawing index
        ddf = - np.sum(meanT_le_0, axis = 2)  # freezing index
        fi = np.sqrt(ddf)/(np.sqrt(ddf) + np.sqrt(ddt)) 

        # now, we will classify the permafrost zones (Reference: GAEZ v4 model documentation: Pg35 -37)
        # for i_row in range(self.im_height):
        #     for i_col in range(self.im_width):
        #         if self.set_mask:
        #             if self.im_mask[i_row, i_col] == self.nodata_val:
        #                 continue         
        #         # Continuous Permafrost Class
        #         if fi[i_row, i_col]> 0.625:
        #             permafrost[i_row, i_col] = 1
                
        #         # Discontinuous Permafrost Class
        #         if fi[i_row, i_col]> 0.57 and fi[i_row, i_col]< 0.625:
        #             permafrost[i_row, i_col] = 2
                
        #         # Sporadic Permafrost Class
        #         if fi[i_row, i_col]> 0.495 and fi[i_row, i_col]< 0.57:
        #             permafrost[i_row, i_col] = 3
                
        #         # No Permafrost Class
        #         if fi[i_row, i_col]< 0.495:
        #             permafrost[i_row, i_col] = 4
        permafrost=np.where(fi>0.625,1,permafrost) # Continuous Permafrost Class  #KLG
        permafrost=np.where((fi>0.57)&(fi<=0.625),2,permafrost) # Discontinuous Permafost Class  #KLG
        permafrost=np.where((fi>0.495)&(fi<=0.57),3,permafrost) # Sporadic Permafrost Class  #KLG
        permafrost=np.where(fi<=0.495,4,permafrost) # No Permafrost Class #KLG

        # to remove the division by zero, the nan values will be converted to zero
        fi = np.nan_to_num(fi)

        if self.set_mask:
            if self.parallel:
                print('in ClimateRegime, computing mask in parallel')                
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask                
            # return [np.where(mask, fi.astype('float32'), np.nan), np.where(mask, permafrost.astype('float32'), np.nan)]  #KLG
            return [np.where(mask, fi.astype('float32'), np.float32(np.nan)), np.where(mask, permafrost.astype('float32'), np.float32(np.nan))]  #KLG
        else:
            return [fi.astype('float32'), permafrost.astype('float32')]  #KLG
        
  

    
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
    
        aez_tclimate = np.zeros((self.im_height, self.im_width), dtype=int)
        # obj_utilities = UtilitiesCalc.UtilitiesCalc()

        for i_r in range(self.im_height):
            for i_c in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_r, i_c] == self.nodata_val:
                        continue

                    else:

                        # tropics highland
                        if tclimate[i_r, i_c] == 1:
                            aez_tclimate[i_r, i_c] = 1

                        elif tclimate[i_r, i_c] == 2:
                            aez_tclimate[i_r, i_c] = 2

                        elif tclimate[i_r, i_c] == 3:
                            aez_tclimate[i_r, i_c] = 3

                        elif tclimate[i_r, i_c] == 4:
                            aez_tclimate[i_r, i_c] = 3

                        elif tclimate[i_r, i_c] == 5:
                            aez_tclimate[i_r, i_c] = 3

                        # grouping all the temperate classes into a single class 4
                        elif tclimate[i_r, i_c] == 6:
                            aez_tclimate[i_r, i_c] = 4

                        elif tclimate[i_r, i_c] == 7:
                            aez_tclimate[i_r, i_c] = 4

                        elif tclimate[i_r, i_c] == 8:
                            aez_tclimate[i_r, i_c] = 4

                        # grouping all the boreal classes into a single class 5
                        elif tclimate[i_r, i_c] == 9:
                            aez_tclimate[i_r, i_c] = 5

                        elif tclimate[i_r, i_c] == 10:
                            aez_tclimate[i_r, i_c] = 5

                        elif tclimate[i_r, i_c] == 11:
                            aez_tclimate[i_r, i_c] = 5

                        # changing the arctic class into class 6
                        elif tclimate[i_r, i_c] == 12:
                            aez_tclimate[i_r, i_c] = 6

        # 2nd Step: Classification of Thermal Zones
        aez_tzone = np.zeros((self.im_height, self.im_width), dtype=int)


        for i_r in range(self.im_height):
            for i_c in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_r, i_c] == self.nodata_val:
                        continue
                    else:
                        mean_temp = np.copy(self.meanT_daily[i_r, i_c, :])
                        # meanT_monthly = obj_utilities.averageDailyToMonthly(mean_temp)
                        meanT_monthly=self.meanT_monthly
                        # one conditional parameter for temperature accumulation
                        temp_acc_10deg = np.copy(self.meanT_daily[i_r, i_c, :])
                        temp_acc_10deg[temp_acc_10deg < 10] = 0

                        # Warm Tzone (TZ1)
                        if np.sum(meanT_monthly >= 10) == 12 and np.mean(mean_temp) >= 20:
                            aez_tzone[i_r, i_c] = 1

                        # Moderately cool Tzone (TZ2)
                        elif np.sum(meanT_monthly >= 5) == 12 and np.sum(meanT_monthly >= 10) >= 8:
                            aez_tzone[i_r, i_c] = 2

                        # TZ3 Moderate
                        elif aez_tclimate[i_r, i_c] == 4 and np.sum(meanT_monthly >= 10) >= 5 and np.sum(mean_temp > 20) >= 75 and np.sum(temp_acc_10deg) > 3000:
                            aez_tzone[i_r, i_c] = 3

                        # TZ4 Cool
                        elif np.sum(meanT_monthly >= 10) >= 4 and np.mean(mean_temp) >= 0:
                            aez_tzone[i_r, i_c] = 4

                        # TZ5 Cold
                        elif np.sum(meanT_monthly >= 10) in range(1, 4) and np.mean(mean_temp) >= 0:
                            aez_tzone[i_r, i_c] = 5

                        # TZ6 Very cold
                        elif np.sum(meanT_monthly < 10) == 12 or np.mean(mean_temp) < 0:
                            aez_tzone[i_r, i_c] = 6

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

        aez_temp_regime = np.zeros((self.im_height, self.im_width), dtype=int)

        for i_r in range(self.im_height):
            for i_c in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_r, i_c] == self.nodata_val:
                        continue
                    else:

                        if aez_tclimate[i_r, i_c] == 1 and aez_tzone[i_r, i_c] == 1:
                            aez_temp_regime[i_r, i_c] = 1  # Tropics, lowland

                        elif aez_tclimate[i_r, i_c] == 2 and aez_tzone[i_r, i_c] in [2, 4]:
                            aez_temp_regime[i_r, i_c] = 2  # Tropics, highland

                        elif aez_tclimate[i_r, i_c] == 3 and aez_tzone[i_r, i_c] == 1:
                            aez_temp_regime[i_r, i_c] = 3  # Subtropics, warm

                        elif aez_tclimate[i_r, i_c] == 3 and aez_tzone[i_r, i_c] == 2:
                            # Subtropics,moderate cool
                            aez_temp_regime[i_r, i_c] = 4

                        elif aez_tclimate[i_r, i_c] == 3 and aez_tzone[i_r, i_c] == 4:
                            aez_temp_regime[i_r, i_c] = 5  # Subtropics,cool

                        elif aez_tclimate[i_r, i_c] == 4 and aez_tzone[i_r, i_c] == 3:
                            # Temperate, moderate
                            aez_temp_regime[i_r, i_c] = 6

                        elif aez_tclimate[i_r, i_c] == 4 and aez_tzone[i_r, i_c] == 4:
                            aez_temp_regime[i_r, i_c] = 7  # Temperate, cool

                        elif aez_tclimate[i_r, i_c] in range(2, 6) and aez_tzone[i_r, i_c] == 5:
                            if np.logical_or(permafrost[i_r, i_c] == 1, permafrost[i_r, i_c] == 2) == False:
                                # Boreal/Cold, no
                                aez_temp_regime[i_r, i_c] = 8
                            else:
                                # Boreal/Cold, with permafrost
                                aez_temp_regime[i_r, i_c] = 9

                        elif aez_tclimate[i_r, i_c] in range(2, 7) and aez_tzone[i_r, i_c] == 6:
                            aez_temp_regime[i_r, i_c] = 10  # Arctic/Very Cold

        # 4th Step: Moisture Regime classes
        # Moisture Regime Class Definition
        # 1 = M1 (desert/arid areas, 0 <= LGP* < 60)
        # 2 = M2 (semi-arid/dry areas, 60 <= LGP* < 180)
        # 3 = M3 (sub-humid/moist areas, 180 <= LGP* < 270)
        # 4 = M4 (humid/wet areas, LGP* >= 270)

        aez_moisture_regime = np.zeros(
            (self.im_height, self.im_width), dtype=int)

        for i_r in range(self.im_height):
            for i_c in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_r, i_c] == self.nodata_val:
                        continue
                    else:

                        # check if LGP t>5 is greater or less than 330 days. If greater, LGP will be used; otherwise, LGP_equv will be used.
                        if lgpt_5[i_r, i_c] > 330:

                            # Class 4 (M4)
                            if lgp[i_r, i_c] >= 270:
                                aez_moisture_regime[i_r, i_c] = 4

                            # Class 3 (M3)
                            elif lgp[i_r, i_c] >= 180 and lgp[i_r, i_c] < 270:
                                aez_moisture_regime[i_r, i_c] = 3

                            # Class 2 (M2)
                            elif lgp[i_r, i_c] >= 60 and lgp[i_r, i_c] < 180:
                                aez_moisture_regime[i_r, i_c] = 2

                            # Class 1 (M1)
                            elif lgp[i_r, i_c] >= 0 and lgp[i_r, i_c] < 60:
                                aez_moisture_regime[i_r, i_c] = 1

                        elif lgpt_5[i_r, i_c] <= 330:

                            # Class 4 (M4)
                            if lgp_equv[i_r, i_c] >= 270:
                                aez_moisture_regime[i_r, i_c] = 4

                            # Class 3 (M3)
                            elif lgp_equv[i_r, i_c] >= 180 and lgp_equv[i_r, i_c] < 270:
                                aez_moisture_regime[i_r, i_c] = 3

                            # Class 2 (M2)
                            elif lgp_equv[i_r, i_c] >= 60 and lgp_equv[i_r, i_c] < 180:
                                aez_moisture_regime[i_r, i_c] = 2

                            # Class 1 (M1)
                            elif lgp_equv[i_r, i_c] >= 0 and lgp_equv[i_r, i_c] < 60:
                                aez_moisture_regime[i_r, i_c] = 1

        # Now, we will classify the agro-ecological zonation
        # By GAEZ v4 Documentation, there are prioritized sequential assignment of AEZ classes in order to ensure the consistency of classification
        aez = np.zeros((self.im_height, self.im_width), dtype=int)

        for i_r in range(self.im_height):
            for i_c in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_r, i_c] == self.nodata_val:
                        continue
                    else:
                        # if it's urban built-up lulc, Dominantly urban/built-up land
                        if soil_terrain_lulc[i_r, i_c] == 8:
                            aez[i_r, i_c] = 56

                        # if it's water/ dominantly water
                        elif soil_terrain_lulc[i_r, i_c] == 7:
                            aez[i_r, i_c] = 57

                        # if it's dominantly very steep terrain/Dominantly very steep terrain
                        elif soil_terrain_lulc[i_r, i_c] == 1:
                            aez[i_r, i_c] = 49

                        # if it's irrigated soils/ Land with ample irrigated soils
                        elif soil_terrain_lulc[i_r, i_c] == 6:
                            aez[i_r, i_c] = 51

                        # if it's hydromorphic soils/ Dominantly hydromorphic soils
                        elif soil_terrain_lulc[i_r, i_c] == 2:
                            aez[i_r, i_c] = 52

                        # Desert/Arid climate
                        elif aez_moisture_regime[i_r, i_c] == 1:
                            aez[i_r, i_c] = 53

                        # BO/Cold climate, with Permafrost
                        elif aez_temp_regime[i_r, i_c] == 9 and aez_moisture_regime[i_r, i_c] in [1, 2, 3, 4] == True:
                            aez[i_r, i_c] = 54

                        # Arctic/ Very cold climate
                        elif aez_temp_regime[i_r, i_c] == 10 and aez_moisture_regime[i_r, i_c] in [1, 2, 3, 4] == True:
                            aez[i_r, i_c] = 55

                        # Severe soil/terrain limitations
                        elif soil_terrain_lulc[i_r, i_c] == 5:
                            aez[i_r, i_c] = 50

                        #######
                        elif aez_temp_regime[i_r, i_c] == 1 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 1

                        elif aez_temp_regime[i_r, i_c] == 1 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 2

                        elif aez_temp_regime[i_r, i_c] == 1 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 3

                        elif aez_temp_regime[i_r, i_c] == 1 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 4

                        elif aez_temp_regime[i_r, i_c] == 1 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 5

                        elif aez_temp_regime[i_r, i_c] == 1 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 6
                        ####
                        elif aez_temp_regime[i_r, i_c] == 2 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 7

                        elif aez_temp_regime[i_r, i_c] == 2 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 8

                        elif aez_temp_regime[i_r, i_c] == 2 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 9

                        elif aez_temp_regime[i_r, i_c] == 2 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 10

                        elif aez_temp_regime[i_r, i_c] == 2 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 11

                        elif aez_temp_regime[i_r, i_c] == 2 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 12
                        ###
                        elif aez_temp_regime[i_r, i_c] == 3 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 13

                        elif aez_temp_regime[i_r, i_c] == 3 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 14

                        elif aez_temp_regime[i_r, i_c] == 3 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 15

                        elif aez_temp_regime[i_r, i_c] == 3 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 16

                        elif aez_temp_regime[i_r, i_c] == 3 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 17

                        elif aez_temp_regime[i_r, i_c] == 3 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 18
                        #####
                        elif aez_temp_regime[i_r, i_c] == 4 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 19

                        elif aez_temp_regime[i_r, i_c] == 4 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 20

                        elif aez_temp_regime[i_r, i_c] == 4 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 21

                        elif aez_temp_regime[i_r, i_c] == 4 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 22

                        elif aez_temp_regime[i_r, i_c] == 4 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 23

                        elif aez_temp_regime[i_r, i_c] == 4 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 24
                        #####
                        elif aez_temp_regime[i_r, i_c] == 5 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 25

                        elif aez_temp_regime[i_r, i_c] == 5 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 26

                        elif aez_temp_regime[i_r, i_c] == 5 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 27

                        elif aez_temp_regime[i_r, i_c] == 5 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 28

                        elif aez_temp_regime[i_r, i_c] == 5 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 29

                        elif aez_temp_regime[i_r, i_c] == 5 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 30
                        ######

                        elif aez_temp_regime[i_r, i_c] == 6 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 31

                        elif aez_temp_regime[i_r, i_c] == 6 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 32

                        elif aez_temp_regime[i_r, i_c] == 6 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 33

                        elif aez_temp_regime[i_r, i_c] == 6 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 34

                        elif aez_temp_regime[i_r, i_c] == 6 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 35

                        elif aez_temp_regime[i_r, i_c] == 6 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 36

                        ###
                        elif aez_temp_regime[i_r, i_c] == 7 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 37

                        elif aez_temp_regime[i_r, i_c] == 7 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 38

                        elif aez_temp_regime[i_r, i_c] == 7 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 39

                        elif aez_temp_regime[i_r, i_c] == 7 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 40

                        elif aez_temp_regime[i_r, i_c] == 7 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 41

                        elif aez_temp_regime[i_r, i_c] == 7 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 42
                        #####

                        elif aez_temp_regime[i_r, i_c] == 8 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 43

                        elif aez_temp_regime[i_r, i_c] == 8 and aez_moisture_regime[i_r, i_c] == 2 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 44

                        elif aez_temp_regime[i_r, i_c] == 8 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 45

                        elif aez_temp_regime[i_r, i_c] == 8 and aez_moisture_regime[i_r, i_c] == 3 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 46

                        elif aez_temp_regime[i_r, i_c] == 8 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 3:
                            aez[i_r, i_c] = 47

                        elif aez_temp_regime[i_r, i_c] == 8 and aez_moisture_regime[i_r, i_c] == 4 and soil_terrain_lulc[i_r, i_c] == 4:
                            aez[i_r, i_c] = 48          

        if self.set_mask:
            return np.where(self.im_mask, aez, np.nan)
        else:        
            return aez
    
    """ 
    Note from Swun: In this code, the logic of temperature amplitude is not added 
    as it brings big discrepency in the temperature regime calculation (in India) 
    compared to previous code. However, the classification schema is now adjusted 
    according to Gunther's agreement and the documentation.
    """
         
    def getMultiCroppingZones(self, t_climate, lgp, lgp_t5, lgp_t10, ts_t0, ts_t10):
    # def getMultiCroppingZones(self, t_climate, lgp, lgp_t5, lgp_t10, ts_t10, ts_t0):
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

        # smoothed daily temperature
        # create a mask of all 1's if the user doesn't provide a mask
        if self.set_mask:
            if self.parallel:
                print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask 
        else:
            mask=np.ones((self.im_height,self.im_width),dtype='int')

        # print(psutil.virtual_memory().free/1E9)
        obj_utilities = UtilitiesCalc.UtilitiesCalc(self.chunk2D,self.chunk3D)  #KLG
        interp_daily_temp=obj_utilities.smoothDailyTemp(self.doy_start,self.doy_end, mask, self.meanT_daily)

        # defining the constant arrays for rainfed and irrigated conditions, all pixel values start with 1
        # multi_crop_rain = np.zeros((self.im_height, self.im_width), dtype = int) # all values started with Zone A
        # multi_crop_irr = np.zeros((self.im_height, self.im_width), dtype = int) # all vauels starts with Zone A
        multi_crop_rain = np.ones((self.im_height, self.im_width), dtype = int) # all values started with Zone A  #KLG
        multi_crop_irr = np.ones((self.im_height, self.im_width), dtype = int) # all vauels starts with Zone A  #KLG
        # ts_g_t5 = np.zeros((self.im_height, self.im_width))
        # ts_g_t10 = np.zeros((self.im_height, self.im_width))          
        
        # Calculation of Accumulated temperature during the growing period at specific temperature thresholds: 5 and 10 degree Celsius
        # interp_meanT_veg_T5=np.where(self.interp_daily_temp>=5.,self.interp_daily_temp,np.nan)  #KLG
        # interp_meanT_veg_T10=np.where(self.interp_daily_temp>=10.,self.interp_daily_temp,np.nan)  #KLG
        interp_meanT_veg_T5=np.where(interp_daily_temp>=5.,interp_daily_temp,np.nan)  #KLG
        interp_meanT_veg_T10=np.where(interp_daily_temp>=10.,interp_daily_temp,np.nan)  #KLG
        ts_g_t5=np.nansum(interp_meanT_veg_T5,axis=2)  #KLG
        ts_g_t10=np.nansum(interp_meanT_veg_T10,axis=2)  #KLG

        # for i_r in range(self.im_height):
        #     for i_c in range(self.im_width):
                
        #         if self.set_mask:
                    
        #             if self.im_mask[i_r, i_c]== self.nodata_val:
        #                 continue
                    
        #             else:
                        
        #                 temp_1D = self.meanT_daily[i_r, i_c, :]
        #                 days = np.arange(0,365)
                        
        #                 deg = 5 # order of polynomical fit
                        
        #                 # creating the function of polyfit
        #                 polyfit = np.poly1d(np.polyfit(days,temp_1D,deg))
                        
        #                 # getting the interpolated value at each DOY
        #                 interp_daily_temp = polyfit(days)
                        
        #                 # Getting the start and end day of vegetative period
        #                 # The crop growth requires minimum temperature of at least 5 deg Celsius
        #                 # If not, the first DOY and the lst DOY of a year will be considered
        #                 try:
        #                     veg_period = days[interp_daily_temp >=5]
        #                     start_veg = veg_period[0]
        #                     end_veg = veg_period[-1]
        #                 except:
        #                     start_veg = 0
        #                     end_veg = 364
                        
        #                 # Slicing the temperature within the vegetative period
        #                 interp_meanT_veg_T5 = interp_daily_temp[start_veg:end_veg]
        #                 interp_meanT_veg_T10 =  interp_daily_temp[start_veg:end_veg] *1
                        
        #                 # Removing the temperature of 5 and 10 deg Celsius thresholds
        #                 interp_meanT_veg_T5[interp_meanT_veg_T5 < 5] = 0
        #                 interp_meanT_veg_T10[interp_meanT_veg_T10 <10] = 0
                        
        #                 # Calculation of Accumulated temperatures during growing period
        #                 ts_g_t5[i_r, i_c] = np.sum(interp_meanT_veg_T5)
        #                 ts_g_t10[i_r, i_c] = np.sum(interp_meanT_veg_T10)

        """Multi cropping zonation for rainfed conditions"""
        multi_crop_rain=np.where((t_climate==1)&(lgp>=360)&(lgp_t5>=360)&(lgp_t10>=360)&(ts_t0>=7200)&(ts_t10>=7000),8,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate==1)&(lgp>=300)&(lgp_t5>=300)&(lgp_t10>=240)&(ts_t0>=7200)&(ts_g_t5>=5100)&(ts_g_t10>=4800)&(multi_crop_rain==1),6,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate==1)&(lgp>=270)&(lgp_t5>=270)&(lgp_t10>=165)&(ts_t0>=5500)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_rain==1),4,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate==1)&(lgp>=240)&(lgp_t5>=240)&(lgp_t10>=165)&(ts_t0>=6400)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_rain==1),4,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate==1)&(lgp>=210)&(lgp_t5>=240)&(lgp_t10>=165)&(ts_t0>=7200)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_rain==1),4,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate==1)&(lgp>=220)&(lgp_t5>=220)&(lgp_t10>=120)&(ts_t0>=5500)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_rain==1),3,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate==1)&(lgp>=200)&(lgp_t5>=200)&(lgp_t10>=120)&(ts_t0>=6400)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_rain==1),3,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate==1)&(lgp>=180)&(lgp_t5>=200)&(lgp_t10>=120)&(ts_t0>=7200)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_rain==1),3,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate==1)&(lgp>=45)&(lgp_t5>=120)&(lgp_t10>=90)&(ts_t0>=1600)&(ts_t10>=1200)&(multi_crop_rain==1),2,multi_crop_rain)   #KLG

        multi_crop_rain=np.where((t_climate!=1)&(lgp>=360)&(lgp_t5>=360)&(lgp_t10>=330)&(ts_t0>=7200)&(ts_t10>=7000)&(multi_crop_rain==1),8,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate!=1)&(lgp>=330)&(lgp_t5>=330)&(lgp_t10>=270)&(ts_t0>=5700)&(ts_t10>=5500)&(multi_crop_rain==1),7,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate!=1)&(lgp>=300)&(lgp_t5>=300)&(lgp_t10>=240)&(ts_t0>=5400)&(ts_t10>=5100)&(ts_g_t5>=5100)&(ts_g_t10>=4800)&(multi_crop_rain==1),6,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate!=1)&(lgp>=240)&(lgp_t5>=270)&(lgp_t10>=180)&(ts_t0>=4800)&(ts_t10>=4500)&(ts_g_t5>=4300)&(ts_g_t10>=4000)&(multi_crop_rain==1),5,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate!=1)&(lgp>=210)&(lgp_t5>=240)&(lgp_t10>=165)&(ts_t0>=4500)&(ts_t10>=3600)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_rain==1),4,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate!=1)&(lgp>=180)&(lgp_t5>=200)&(lgp_t10>=120)&(ts_t0>=3600)&(ts_t10>=3000)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_rain==1),3,multi_crop_rain)  #KLG
        multi_crop_rain=np.where((t_climate!=1)&(lgp>=45)&(lgp_t5>=120)&(lgp_t10>=90)&(ts_t0>=1600)&(ts_t10>=1200)&(multi_crop_rain==1),2,multi_crop_rain)  #KLG
                                                     
        # for i_r in range(self.im_height):
        #     for i_c in range(self.im_width):
                
        #         if self.set_mask:
                    
        #             if self.im_mask[i_r, i_c]== self.nodata_val:
        #                 continue
                    
        #             else:
                        
        #                 if t_climate[i_r, i_c]== 1:
                            
        #                     if np.all([lgp[i_r, i_c]>=360, lgp_t5[i_r, i_c]>=360, lgp_t10[i_r, i_c]>=360, ts_t0[i_r, i_c]>=7200, ts_t10[i_r, i_c]>=7000])== True:
        #                         multi_crop_rain[i_r, i_c] = 8
                            
        #                     elif np.all([lgp[i_r, i_c]>=300, lgp_t5[i_r, i_c]>=300, lgp_t10[i_r, i_c]>=240, ts_t0[i_r, i_c]>=7200, ts_g_t5[i_r, i_c]>=5100, ts_g_t10[i_r, i_c]>=4800])== True:
        #                         multi_crop_rain[i_r, i_c] = 6
                            
        #                     elif np.all([lgp[i_r, i_c]>=270, lgp_t5[i_r, i_c]>=270, lgp_t10[i_r, i_c]>=165, ts_t0[i_r, i_c]>=5500, ts_g_t5[i_r, i_c]>=4000, ts_g_t10[i_r, i_c]>=3200])== True:
        #                         multi_crop_rain[i_r, i_c] = 4 # Ok
                                
        #                     elif np.all([lgp[i_r, i_c]>=240, lgp_t5[i_r, i_c]>=240, lgp_t10[i_r, i_c]>=165, ts_t0[i_r, i_c]>=6400, ts_g_t5[i_r, i_c]>=4000, ts_g_t10[i_r, i_c]>=3200])== True:
        #                         multi_crop_rain[i_r, i_c] = 4 # Ok
                            
        #                     elif np.all([lgp[i_r, i_c]>=210, lgp_t5[i_r, i_c]>=240, lgp_t10[i_r, i_c]>=165, ts_t0[i_r, i_c]>=7200, ts_g_t5[i_r, i_c]>=4000, ts_g_t10[i_r, i_c]>=3200])== True:
        #                         multi_crop_rain[i_r, i_c] = 4 # OK
                            
        #                     elif np.all([lgp[i_r, i_c]>=220, lgp_t5[i_r, i_c]>=220, lgp_t10[i_r, i_c]>=120, ts_t0[i_r, i_c]>=5500, ts_g_t5[i_r, i_c]>=3200, ts_g_t10[i_r, i_c]>=2700])== True:
        #                         multi_crop_rain[i_r, i_c] = 3 #OK
                            
        #                     elif np.all([lgp[i_r, i_c]>=200, lgp_t5[i_r, i_c]>=200, lgp_t10[i_r, i_c]>=120, ts_t0[i_r, i_c]>=6400, ts_g_t5[i_r, i_c]>=3200, ts_g_t10[i_r, i_c]>=2700])== True:
        #                         multi_crop_rain[i_r, i_c] = 3# OK
                            
        #                     elif np.all([lgp[i_r, i_c]>=180, lgp_t5[i_r, i_c]>=200, lgp_t10[i_r, i_c]>=120, ts_t0[i_r, i_c]>=7200, ts_g_t5[i_r, i_c]>=3200, ts_g_t10[i_r, i_c]>=2700])== True:
        #                         multi_crop_rain[i_r, i_c] = 3 # OK
                            
        #                     elif np.all([lgp[i_r, i_c]>=45, lgp_t5[i_r, i_c]>=120, lgp_t10[i_r, i_c]>=90, ts_t0[i_r, i_c]>=1600, ts_t10[i_r, i_c]>=1200]) == True:
        #                         multi_crop_rain[i_r, i_c] = 2 # Ok
                                
        #                     else:
        #                         multi_crop_rain[i_r, i_c] = 1 # Ok
                            
        #                 elif t_climate[i_r, i_c] != 1:
                            
        #                     if np.all([lgp[i_r, i_c]>=360, lgp_t5[i_r, i_c]>=360, lgp_t10[i_r, i_c]>=330, ts_t0[i_r, i_c]>=7200, ts_t10[i_r, i_c]>=7000])== True:
        #                         multi_crop_rain[i_r, i_c] = 8 # Ok
                            
        #                     elif np.all([lgp[i_r, i_c]>=330, lgp_t5[i_r, i_c]>=330, lgp_t10[i_r, i_c]>=270, ts_t0[i_r, i_c]>=5700, ts_t10[i_r, i_c]>=5500])== True:
        #                         multi_crop_rain[i_r, i_c] = 7 # Ok
                            
        #                     elif np.all([lgp[i_r, i_c]>=300, lgp_t5[i_r, i_c]>=300, lgp_t10[i_r, i_c]>=240, ts_t0[i_r, i_c]>=5400, ts_t10[i_r, i_c]>=5100, ts_g_t5[i_r, i_c]>=5100, ts_g_t10[i_r, i_c]>=4800])== True:
        #                         multi_crop_rain[i_r, i_c] = 6 # Ok
                            
        #                     elif np.all([lgp[i_r, i_c]>=240, lgp_t5[i_r, i_c]>=270, lgp_t10[i_r, i_c]>=180, ts_t0[i_r, i_c]>=4800, ts_t10[i_r, i_c]>=4500, ts_g_t5[i_r, i_c]>=4300, ts_g_t10[i_r, i_c]>=4000])== True:
        #                         multi_crop_rain[i_r, i_c] = 5 # Ok
                            
        #                     elif np.all([lgp[i_r, i_c]>=210, lgp_t5[i_r, i_c]>=240, lgp_t10[i_r, i_c]>=165, ts_t0[i_r, i_c]>=4500, ts_t10[i_r, i_c]>=3600, ts_g_t5[i_r, i_c]>=4000, ts_g_t10[i_r, i_c]>=3200])== True:
        #                         multi_crop_rain[i_r, i_c] = 4 #OK
                            
        #                     elif np.all([lgp[i_r, i_c]>=180, lgp_t5[i_r, i_c]>=200, lgp_t10[i_r, i_c]>=120, ts_t0[i_r, i_c]>=3600, ts_t10[i_r, i_c]>=3000, ts_g_t5[i_r, i_c]>=3200, ts_g_t10[i_r, i_c]>=2700])== True:
        #                         multi_crop_rain[i_r, i_c] = 3 # Ok
                            
        #                     elif np.all([lgp[i_r, i_c]>=45, lgp_t5[i_r, i_c]>=120, lgp_t10[i_r, i_c]>=90, ts_t0[i_r, i_c]>=1600, ts_t10[i_r, i_c]>=1200]) == True:
        #                         multi_crop_rain[i_r, i_c] = 2 #Ok
                            
        #                     else:
        #                         multi_crop_rain[i_r, i_c] = 1 #Ok
                            
        
        """Multi cropping zonation for irrigated conditions"""
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=360)&(lgp_t10>=360)&(ts_t0>=7200)&(ts_t10>=7000),8,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=300)&(lgp_t10>=240)&(ts_t0>=7200)&(ts_g_t5>=5100)&(ts_g_t10>=4800)&(multi_crop_irr==1),6,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=270)&(lgp_t10>=165)&(ts_t0>=5500)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_rain==1),4,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=240)&(lgp_t10>=165)&(ts_t0>=6400)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_rain==1),4,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=240)&(lgp_t10>=165)&(ts_t0>=7200)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_rain==1),4,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=220)&(lgp_t10>=120)&(ts_t0>=5500)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_rain==1),3,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=200)&(lgp_t10>=120)&(ts_t0>=6400)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_rain==1),3,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=200)&(lgp_t10>=120)&(ts_t0>=7200)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_rain==1),3,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate==1)&(lgp_t5>=120)&(lgp_t10>=90)&(ts_t0>=1600)&(ts_t10>=1200)&(multi_crop_rain==1),2,multi_crop_irr)  #KLG

        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=360)&(lgp_t10>=330)&(ts_t0>=7200)&(ts_t10>=7000)&(multi_crop_rain==1),8,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=330)&(lgp_t10>=270)&(ts_t0>=5700)&(ts_t10>=5500)&(multi_crop_rain==1),7,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=300)&(lgp_t10>=240)&(ts_t0>=5400)&(ts_t10>=5100)&(ts_g_t5>=5100)&(ts_g_t10>=4800)&(multi_crop_rain==1),6,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=270)&(lgp_t10>=180)&(ts_t0>=4800)&(ts_t10>=4500)&(ts_g_t5>=4300)&(ts_g_t10>=4000)&(multi_crop_rain==1),5,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=240)&(lgp_t10>=165)&(ts_t0>=4500)&(ts_t10>=3600)&(ts_g_t5>=4000)&(ts_g_t10>=3200)&(multi_crop_rain==1),4,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=200)&(lgp_t10>=120)&(ts_t0>=3600)&(ts_t10>=3000)&(ts_g_t5>=3200)&(ts_g_t10>=2700)&(multi_crop_rain==1),3,multi_crop_irr)  #KLG
        multi_crop_irr=np.where((t_climate!=1)&(lgp_t5>=120)&(lgp_t10>=90)&(ts_t0>=1600)&(ts_t10>=1200)&(multi_crop_rain==1),2,multi_crop_irr)  #KLG

        multi_crop_rain=multi_crop_rain.compute()
        multi_crop_irr=multi_crop_irr.compute()
        
        # for i_r in range(self.im_height):
        #     for i_c in range(self.im_width):
                
        #         if self.set_mask:
                    
        #             if self.im_mask[i_r, i_c]== self.nodata_val:
        #                 continue
                    
        #             else:
                        
        #                 if t_climate[i_r, i_c]== 1:
                            
        #                     if np.all([lgp_t5[i_r, i_c]>=360, lgp_t10[i_r, i_c]>=360, ts_t0[i_r, i_c]>=7200, ts_t10[i_r, i_c]>=7000])==True:
        #                         multi_crop_irr[i_r, i_c] =8 # ok
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=300, lgp_t10[i_r, i_c]>=240, ts_t0[i_r, i_c]>=7200, ts_g_t5[i_r, i_c]>=5100, ts_g_t10[i_r, i_c]>=4800])==True:
        #                         multi_crop_irr[i_r, i_c] =6 # ok
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=270, lgp_t10[i_r, i_c]>=165, ts_t0[i_r, i_c]>=5500, ts_g_t5[i_r, i_c]>=4000, ts_g_t10[i_r, i_c]>=3200]) == True:
        #                         multi_crop_irr[i_r, i_c] =4 # Ok
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=240, lgp_t10[i_r, i_c]>=165, ts_t0[i_r, i_c]>=6400, ts_g_t5[i_r, i_c]>=4000, ts_g_t10[i_r, i_c]>=3200])== True:
        #                         multi_crop_irr[i_r, i_c] =4 #ok
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=240, lgp_t10[i_r, i_c]>=165, ts_t0[i_r, i_c]>=7200, ts_g_t5[i_r, i_c]>=4000, ts_g_t10[i_r, i_c]>=3200])== True:
        #                         multi_crop_irr[i_r, i_c] =4 # ok
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=220, lgp_t10[i_r, i_c]>=120, ts_t0[i_r, i_c]>=5500, ts_g_t5[i_r, i_c]>=3200, ts_g_t10[i_r, i_c]>=2700]) == True:
        #                         multi_crop_irr[i_r, i_c] =3 #Ok
                                
        #                     elif np.all([lgp_t5[i_r, i_c]>=200, lgp_t10[i_r, i_c]>=120, ts_t0[i_r, i_c]>=6400, ts_g_t5[i_r, i_c]>=3200, ts_g_t10[i_r, i_c]>=2700])== True:
        #                         multi_crop_irr[i_r, i_c] =3 #ok
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=200, lgp_t10[i_r, i_c]>=120, ts_t0[i_r, i_c]>=7200, ts_g_t5[i_r, i_c]>=3200, ts_g_t10[i_r, i_c]>=2700])==True:
        #                         multi_crop_irr[i_r, i_c] =3 # Ok
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=120, lgp_t10[i_r, i_c]>=90, ts_t0[i_r, i_c]>=1600, ts_t10[i_r, i_c]>=1200]) == True:
        #                         multi_crop_irr[i_r, i_c] =2 # Ok
                            
        #                     else:
        #                         multi_crop_irr[i_r, i_c] =1 # Ok
                        
        #                 elif t_climate[i_r, i_c] != 1:
                            
        #                     if np.all([lgp_t5[i_r, i_c]>=360, lgp_t10[i_r, i_c]>=330, ts_t0[i_r, i_c]>=7200, ts_t10[i_r, i_c]>=7000])==True:
        #                         multi_crop_irr[i_r, i_c] = 8
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=330, lgp_t10[i_r, i_c]>=270, ts_t0[i_r, i_c]>=5700, ts_t10[i_r, i_c]>=5500])==True:
        #                         multi_crop_irr[i_r, i_c] = 7 # ok
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=300, lgp_t10[i_r, i_c]>=240, ts_t0[i_r, i_c]>=5400, ts_t10[i_r, i_c]>=5100, ts_g_t5[i_r, i_c]>=5100, ts_g_t10[i_r, i_c]>=4800])==True:
        #                         multi_crop_irr[i_r, i_c] = 6 #ok
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=270, lgp_t10[i_r, i_c]>=180, ts_t0[i_r, i_c]>=4800, ts_t10[i_r, i_c]>=4500, ts_g_t5[i_r, i_c]>=4300, ts_g_t10[i_r, i_c]>=4000])==True:
        #                         multi_crop_irr[i_r, i_c] = 5 #ok
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=240, lgp_t10[i_r, i_c]>=165, ts_t0[i_r, i_c]>=4500, ts_t10[i_r, i_c]>=3600, ts_g_t5[i_r, i_c]>=4000, ts_g_t10[i_r, i_c]>=3200])==True:
        #                         multi_crop_irr[i_r, i_c] = 4 #ok
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=200, lgp_t10[i_r, i_c]>=120, ts_t0[i_r, i_c]>=3600, ts_t10[i_r, i_c]>=3000, ts_g_t5[i_r, i_c]>=3200, ts_g_t10[i_r, i_c]>=2700])==True:
        #                         multi_crop_irr[i_r, i_c] = 3 # ok
                            
        #                     elif np.all([lgp_t5[i_r, i_c]>=120, lgp_t10[i_r, i_c]>=90, ts_t0[i_r, i_c]>=1600, ts_t10[i_r, i_c]>=1200])==True:
        #                         multi_crop_irr[i_r, i_c] = 2 #ok
                            
        #                     else:
        #                         multi_crop_irr[i_r, i_c] = 1

        if self.set_mask:
            if self.parallel:
                print('in ClimateRegime, computing mask in parallel')
                mask=self.im_mask.compute()
            else:
                mask=self.im_mask
            return [np.where(mask, multi_crop_rain.astype('float32'), np.nan), np.where(mask, multi_crop_irr.astype('float32'), np.float32(np.nan))]  #KLG
        else:        
            return [multi_crop_rain.astype('float32'), multi_crop_irr.astype('float32')]  #KLG
                        
    

#----------------- End of file -------------------------#