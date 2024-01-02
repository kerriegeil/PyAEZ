"""
PyAEZ version 2.1.0 (June 2023)
This CropSimulation Class simulates all the possible crop cycles to find 
the best crop cycle that produces maximum yield for a particular grid
2020: N. Lakmal Deshapriya
2022/2023: Swun Wunna Htet, Kittiphon Boonma
"""

import numpy as np
import pandas as pd
try:
    import gdal
except:
    from osgeo import gdal

# from pyaez import UtilitiesCalc,BioMassCalc,ETOCalc,CropWatCalc,ThermalScreening,LGPCalc
import UtilitiesCalc as UtilitiesCalc  
import ETOCalc as ETOCalc
import LGPCalc as LGPCalc
import BioMassCalc as BioMassCalc
import CropWatCalc as CropWatCalc
import ThermalScreening as ThermalScreening


class CropSimulation(object):

    def __init__(self):
        """Initiate a Class instance
        """        
        self.set_mask = False
        self.set_tclimate_screening = False
        self.set_lgpt_screening = False
        self.set_Tsum_screening = False
        self.set_Permafrost_screening = False  
        self.set_adjustment = False 
        self.setTypeBConstraint = False
        self.set_monthly = False


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


    # def setMonthlyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):
    #     """Load MONTHLY climate data into the Class and calculate the Reference Evapotranspiration (ETo)

    #     Args:
    #         min_temp (3D NumPy): Monthly minimum temperature [Celcius]
    #         max_temp (3D NumPy): Monthly maximum temperature [Celcius]
    #         precipitation (3D NumPy): Monthly total precipitation [mm/day]
    #         short_rad (3D NumPy): Monthly solar radiation [W/m2]
    #         wind_speed (3D NumPy): Monthly windspeed at 2m altitude [m/s]
    #         rel_humidity (3D NumPy): Monthly relative humidity [percentage decimal, 0-1]
    #     """
    #     rel_humidity[rel_humidity > 0.99] = 0.99
    #     rel_humidity[rel_humidity < 0.05] = 0.05
    #     short_rad[short_rad < 0] = 0
    #     wind_speed[wind_speed < 0] = 0
    #     self.minT_monthly = min_temp
    #     self.maxT_monthly = max_temp
    #     self.totalPrec_monthly = precipitation
    #     self.shortRad_monthly = short_rad
    #     self.wind2m_monthly = wind_speed
    #     self.rel_humidity_monthly = rel_humidity
    #     self.meanT_daily = np.zeros((self.im_height, self.im_width, 365))
    #     self.totalPrec_daily = np.zeros((self.im_height, self.im_width, 365))
    #     self.pet_daily = np.zeros((self.im_height, self.im_width, 365))
    #     self.minT_daily = np.zeros((self.im_height, self.im_width, 365))
    #     self.maxT_daily = np.zeros((self.im_height, self.im_width, 365))

    #     # Interpolate monthly to daily data
    #     obj_utilities = UtilitiesCalc.UtilitiesCalc()

    #     self.meanT_monthly = 0.5*(min_temp+max_temp)

    #     for i_row in range(self.im_height):
    #         for i_col in range(self.im_width):

    #             if self.set_mask:
    #                 if self.im_mask[i_row, i_col] == self.nodata_val:
    #                     continue

    #             self.meanT_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(
    #                 self.meanT_monthly[i_row, i_col, :], 1, 365)
    #             self.totalPrec_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(
    #                 precipitation[i_row, i_col, :], 1, 365, no_minus_values=True)
    #             self.minT_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(
    #                 min_temp[i_row, i_col, :], 1, 365)
    #             self.maxT_daily[i_row, i_col, :] = obj_utilities.interpMonthlyToDaily(
    #                 max_temp[i_row, i_col, :], 1, 365)
    #             radiation_daily = obj_utilities.interpMonthlyToDaily(
    #                 short_rad[i_row, i_col, :], 1, 365, no_minus_values=True)
    #             wind_daily = obj_utilities.interpMonthlyToDaily(
    #                 wind_speed[i_row, i_col, :], 1, 365, no_minus_values=True)
    #             rel_humidity_daily = obj_utilities.interpMonthlyToDaily(
    #                 rel_humidity[i_row, i_col, :], 1, 365, no_minus_values=True)

    #             # calculation of reference evapotranspiration (ETo)
    #             obj_eto = ETOCalc.ETOCalc(
    #                 1, 365, self.latitude[i_row, i_col], self.elevation[i_row, i_col])
    #             # convert w/m2 to MJ/m2/day
    #             shortrad_daily_MJm2day = (radiation_daily*3600*24)/1000000
    #             obj_eto.setClimateData(
    #                 self.minT_daily[i_row, i_col, :], self.maxT_daily[i_row, i_col, :], wind_daily, shortrad_daily_MJm2day, rel_humidity_daily)
    #             self.pet_daily[i_row, i_col, :] = obj_eto.calculateETO()

    #     # Sea-level adjusted mean temperature
    #     self.meanT_daily_sealevel = self.meanT_daily + \
    #         np.tile(np.reshape(self.elevation/100*0.55,
    #                 (self.im_height, self.im_width, 1)), (1, 1, 365))
    #     # P over PET ratio(to eliminate nan in the result, nan is replaced with zero)
    #     self.P_by_PET_daily = np.divide(
    #         self.totalPrec_daily, self.pet_daily, out=np.zeros_like(self.totalPrec_daily), where=(self.pet_daily != 0))

    #     self.set_monthly=True

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
            short_rad = short_rad*0.0864#(3600.*24./1000000.)# convert units
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
            meanT_daily_sealevel = self.meanT_daily + (da.broadcast_to(self.elevation[:,:,np.newaxis],(self.im_height,self.im_width,self.doy_end))/55)
            # print('in ClimateRegime, agg daily to meanT_monthly_sealevel in parallel')
            self.meanT_monthly_sealevel = obj_utilities.averageDailyToMonthly(meanT_daily_sealevel)   
            del meanT_daily_sealevel
            if self.set_mask:
                mask_monthly=da.broadcast_to(self.im_mask[:,:,np.newaxis],(self.im_height,self.im_width,12))
                self.meanT_monthly_sealevel = da.where(mask_monthly,self.meanT_monthly_sealevel,np.float32(np.nan)).compute()
            
            ### CALCULATE P_BY_PET_MONTHLY ####
            # same for P_by_PET_monthly, we only use monthly later
            pr=precipitation.rechunk(chunks=self.chunk3D) # dask array
            pet=da.from_array(self.pet_daily,chunks=self.chunk3D)  # dask array
            with np.errstate(divide='ignore', invalid='ignore'):
                P_by_PET_daily = np.nan_to_num(pr/pet)  #dask array
                P_by_PET_monthly = obj_utilities.averageDailyToMonthly(P_by_PET_daily)  # compute monthly values (np array)

            if self.set_mask:
                mask_monthly=mask_monthly.compute() # numpy array
                self.P_by_PET_monthly = np.where(mask_monthly,P_by_PET_monthly,np.float32(np.nan)) # implement mask
            else:
                self.P_by_PET_monthly=P_by_PET_monthly
            del pr, pet, P_by_PET_daily,P_by_PET_monthly

        else:
            ### CALCULATE MEANT_MONTHLY_SEALEVEL ###
            meanT_daily_sealevel = self.meanT_daily + np.expand_dims(self.elevation/55,axis=2)      
            self.meanT_monthly_sealevel = obj_utilities.averageDailyToMonthly(meanT_daily_sealevel)   
            del meanT_daily_sealevel          

            ### CALCULATE P_BY_PET_MONTHLY ####
            with np.errstate(invalid='ignore',divide='ignore'):
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

    
    # def setLocationTerrainData(self, lats, elevation):  
        # option to take all lats as an input  
        # why not take in an array of all latitudes here instead of regenerating lats from min/max
        # could avoid slight shifts/precision problems with the grid
    def setLocationTerrainData(self, lat_min, lat_max, location, elevation): 
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


    # For this function, we need to explain how to set up excel sheet in the User Guide (Important)
    def readCropandCropCycleParameters(self, file_path, crop_name):
        """
        Mandatory function to import the excel sheet of crop-specific parameters,
        crop water requirements, management info, perennial adjustment parameters,
        and TSUM screening thresholds.

        Parameters
        ----------
        file_path : String.
            The file path of the external excel sheet in xlsx.
        crop_name : String.
            Unique name of crop for crop simulation.

        Returns
        -------
        None.

        """

        self.crop_name = crop_name
        df = pd.read_excel(file_path)

        crop_df_index = df.index[df['Crop_name'] == crop_name].tolist()[0]
        crop_df = df.loc[df['Crop_name'] == crop_name]

        self.setCropParameters(LAI=crop_df['LAI'][crop_df_index], HI=crop_df['HI'][crop_df_index], legume=crop_df['legume'][crop_df_index], adaptability=int(crop_df['adaptability'][crop_df_index]), cycle_len=int(crop_df['cycle_len'][crop_df_index]), D1=crop_df['D1']
                               [crop_df_index], D2=crop_df['D2'][crop_df_index], min_temp=crop_df['min_temp'][crop_df_index], aLAI=crop_df['aLAI'][crop_df_index], bLAI=crop_df['bLAI'][crop_df_index], aHI=crop_df['aHI'][crop_df_index], bHI=crop_df['bHI'][crop_df_index])
        self.setCropCycleParameters(stage_per=[crop_df['stage_per_1'][crop_df_index], crop_df['stage_per_2'][crop_df_index], crop_df['stage_per_3'][crop_df_index], crop_df['stage_per_4'][crop_df_index]], kc=[crop_df['kc_0'][crop_df_index], crop_df['kc_1'][crop_df_index], crop_df['kc_2']
                                    [crop_df_index]], kc_all=crop_df['kc_all'][crop_df_index], yloss_f=[crop_df['yloss_f0'][crop_df_index], crop_df['yloss_f1'][crop_df_index], crop_df['yloss_f2'][crop_df_index], crop_df['yloss_f3'][crop_df_index]], yloss_f_all=crop_df['yloss_f_all'][crop_df_index])

        # perennial = 1, annual = 0
        if crop_df['annual/perennial flag'][crop_df_index] == 1:
            self.perennial = True
        else:
            self.perennial = False

        # If users provide all TSUM thresholds, TSUM screening
        if np.all([crop_df['LnS'][crop_df_index] != np.nan, crop_df['LsO'][crop_df_index] != np.nan, crop_df['LO'][crop_df_index] != np.nan, crop_df['HnS'][crop_df_index] != np.nan, crop_df['HsO'][crop_df_index] != np.nan, crop_df['HO'][crop_df_index] != np.nan]):
            self.setTSumScreening(LnS=crop_df['LnS'][crop_df_index], LsO=crop_df['LsO'][crop_df_index], LO=crop_df['LO'][crop_df_index],
                                  HnS=crop_df['HnS'][crop_df_index], HsO=crop_df['HsO'][crop_df_index], HO=crop_df['HO'][crop_df_index])

        # releasing memory
        del (crop_df_index, crop_df)


    def setSoilWaterParameters(self, Sa, pc):
        """This function allow user to set up the parameters related to the soil water storage.

        Args:
            Sa (float or 2D numpy): Available  soil moisture holding capacity
            pc (float): Soil water depletion fraction below which ETa<ETo
        """        
        self.Sa = Sa  # available soil moisture holding capacity (mm/m) , assumption
        self.pc = pc  # soil water depletion fraction below which ETa < ETo (from literature)

    
    
    """Supporting functions nested within the mandatory functions"""

    def setCropParameters(self, LAI, HI, legume, adaptability, cycle_len, D1, D2, min_temp, aLAI, bLAI, aHI, bHI):
        """This function allows users to set up the main crop parameters necessary for PyAEZ.

        Args:
            LAI (float): Leaf Area Index
            HI (float): Harvest Index
            legume (binary, yes=1, no=0): Is the crop legume?
            adaptability (int): Crop adaptability clases (1-4)
            cycle_len (int): Length of crop cycle
            D1 (float): Rooting depth at the beginning of the crop cycle [m]
            D2 (float): Rooting depth after crop maturity [m]
        """
        self.LAi = LAI  # leaf area index
        self.HI = HI  # harvest index
        self.legume = legume  # binary value
        self.adaptability = adaptability  # one of [1,2,3,4] classes
        self.cycle_len = cycle_len  # length of growing period
        self.D1 = D1  # rooting depth 1 (m)
        self.D2 = D2  # rooting depth 2 (m)
        self.min_temp = min_temp  # minimum temperature
        self.aLAI = aLAI
        self.bLAI = bLAI
        self.aHI = aHI
        self.bHI = bHI

    def setCropCycleParameters(self, stage_per, kc, kc_all, yloss_f, yloss_f_all):
        self.d_per = stage_per  # Percentage for D1, D2, D3, D4 stages
        self.kc = kc  # 3 crop water requirements for initial, reproductive, the end of the maturation stages
        self.kc_all = kc_all  # crop water requirements for entire growth cycle
        self.yloss_f = yloss_f  # yield loss for D1, D2, D3, D4
        self.yloss_f_all = yloss_f_all  # yield loss for entire growth cycle

    
    
    '''Additional optional functions'''

    # set mask of study area, this is optional
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
        meanT=meanT_monthly.mean(axis=2)   
        nmo_ge_10C=(meanT_monthly_sealevel >= 10).sum(axis=2)   
        
        if self.chunk3D:
            # print('in ClimateRegime, computing latitude in parallel')
            latitude=self.latitude.compute()
        else:
            latitude=self.latitude

        # Tropics   
        # Tropical lowland   
        thermal_climate=np.where((min_sealev_meanT>=18.) & (Ta_diff<15.) & (meanT>=20.),1,thermal_climate)   
        # Tropical highland   
        thermal_climate=np.where((min_sealev_meanT>=18.) & (Ta_diff<15.) & (meanT<20.) & (thermal_climate==0),2,thermal_climate)   
        
        # SubTropic   
        # Subtropics Low Rainfall   
        thermal_climate=np.where((min_sealev_meanT>=5.) & (nmo_ge_10C>=8) & (prsum<250) & (thermal_climate==0),3,thermal_climate)   
        # Subtropics Summer Rainfall   
        thermal_climate=np.where((min_sealev_meanT>=5.) & (nmo_ge_10C>=8) & (prsum>=250)& (latitude>=0) & (summer_PET0>=winter_PET0) & (thermal_climate==0),4,thermal_climate)   
        thermal_climate=np.where((min_sealev_meanT>=5.) & (nmo_ge_10C>=8) & (prsum>=250)& (latitude<0) & (summer_PET0<winter_PET0) & (thermal_climate==0),4,thermal_climate)   
        # Subtropics Winter Rainfall   
        thermal_climate=np.where((min_sealev_meanT>=5.) & (nmo_ge_10C>=8) & (prsum>=250)& (latitude>=0) & (summer_PET0<winter_PET0) & (thermal_climate==0),5,thermal_climate)   
        thermal_climate=np.where((min_sealev_meanT>=5.) & (nmo_ge_10C>=8) & (prsum>=250)& (latitude<0) & (summer_PET0>=winter_PET0) & (thermal_climate==0),5,thermal_climate)   
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
            return thermal_climate

    
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
                mask_np=mask_c.blocks[0,i].compute()
                Tx365_np=Tx365_c.blocks[0,i,0].compute()
                islgp_np=islgp_c.blocks[0,i,0].compute()
                Pcp365_np=Pcp365_c.blocks[0,i,0].compute()
                Pet365_np=Pet365_c.blocks[0,i,0].compute()
                Wb_old_np=Wb_old_c.blocks[0,i].compute()
                Sb_old_np=Sb_old_c.blocks[0,i].compute()
                istart0_np=istart0_c.blocks[0,i].compute()
                istart1_np=istart1_c.blocks[0,i].compute()
                lgpt5_np=lgpt5_c.blocks[0,i].compute()
                eta_class_np=eta_class_c.blocks[0,i].compute()
                p_np=p_c.blocks[0,i,0].compute()

                # compute lgp_tot in chunks
                results.append(LGPCalc.EtaCalc(mask_np,Tx365_np,islgp_np,Pcp365_np,\
                                            Txsnm,Fsnm,Pet365_np,Wb_old_np,Sb_old_np,\
                                            istart0_np,istart1_np,Sa,D,p_np,kc_list,\
                                            lgpt5_np,eta_class_np,self.doy_start,self.doy_end,self.parallel))
            # task_time=timer()-start
            # print('time spent in compute',task_time)  

            del self.pet_daily # free up RAM

            # concatenate result chunks
            # start=timer()
            lgp_tot=np.concatenate(results,axis=1)
            # task_time=timer()-start
            # print('time spent on lgp_tot concat',task_time)   

            if self.set_mask:
                return np.where(self.im_mask.compute(), lgp_tot.astype('float32'), np.float32(np.nan))   
            else:
                return lgp_tot.astype('float32')   
        else:
            try:
                istart0,istart1=LGPCalc.rainPeak(self.meanT_daily,self.lgpt5)
                ng=np.zeros(self.pet_daily.shape,dtype='float32')
                p = LGPCalc.psh(ng,self.pet_daily)

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


    def ImportLGPandLGPTforPerennial(self, lgp, lgpt5, lgpt10):
        """
        Mandatory step of input data required for perennial crop simulation.
        This function is run before the actual crop simulation.

        Parameters
        ----------
        lgp : 2-D numpy array
            Length of Growing Period.
        lgpt5 : 2-D numpy array
            Temperature Growing Period at 5℃ threshold.
        lgpt10 : 2-D numpy array
            Temperature Growing Period at 10℃ threshold.

        Returns
        -------
        None.

        """
        self.LGP = lgp
        self.LGPT5 = lgpt5
        self.LGPT10 = lgpt10

    def adjustForPerennialCrop(self,  cycle_len, aLAI, bLAI, aHI, bHI, rain_or_irr):
        """If a perennial crop is introduced, PyAEZ will perform adjustment 
        on the Leaf Area Index (LAI) and the Harvest Index (HI) based 
        on the effective growing period.

        Args:
            aLAI (int): alpha coefficient for LAI
            bLAI (int): beta coefficient for LAI
            aHI (int): alpha coefficient for HI
            bHI (int): beta coefficient for HI
        """        
        if rain_or_irr == 'rain':
            # leaf area index adjustment for perennial crops
            self.LAi_rain = self.LAi * ((cycle_len-aLAI)/bLAI)
            # harvest index adjustment for perennial crops
            self.HI_rain = self.HI * ((cycle_len-aHI)/bHI)

        if rain_or_irr == 'irr':
            # leaf area index adjustment for perennial crops
            self.LAi_irr = self.LAi * ((cycle_len-aLAI)/bLAI)
            # harvest index adjustment for perennial crops
            self.HI_irr = self.HI * ((cycle_len-aHI)/bHI)

    """ Thermal Screening functions (Optional)"""

    def setThermalClimateScreening(self, t_climate, no_t_climate):
        """
        The thermal screening function omit out user-specified thermal climate classes
        not suitable for a particular crop for crop simulation. Using this optional 
        function will activate application of thermal climate screening in crop cycle simulation.
    

        Parameters
        ----------
        t_climate : 2-D numpy array
            Thermal Climate.
        no_t_climate : list
            A list of thermal climate classes not suitable for crop simulation.

        Returns
        -------
        None.

        """
        self.t_climate = t_climate
        self.no_t_climate = no_t_climate  # list of unsuitable thermal climate

        self.set_tclimate_screening = True

    # set suitability screening, this is also optional
    def setLGPTScreening(self, no_lgpt, optm_lgpt):
        """Set screening parameters for thermal growing period (LGPt)

        Args:
            no_lgpt (3-item list): 3 'not suitable' LGPt conditions
            optm_lgpt (3-item list): 3 'optimum' LGPt conditions
        """        
        self.no_lgpt = no_lgpt
        self.optm_lgpt = optm_lgpt

        self.set_lgpt_screening = True

    def setTSumScreening(self, LnS, LsO, LO, HnS, HsO, HO):
        """
        This thermal screening corresponds to Type A constraint (TSUM Screeing) of GAEZ which
        uses six TSUM thresholds for optimal, sub-optimal and not suitable conditions. Using 
        this optional function will activate application of TSUM screening in crop cycle simulation.
        

        Parameters
        ----------
        LnS : Integer
            Lower boundary of not-suitable accumulated heat unit range.
        LsO : Integer
            Lower boundary of sub-optimal accumulated heat unit range.
        LO : Integer
            Lower boundary of optimal accumulated heat unit range.
        HnS : Integer
            Upper boundary of not-suitable accumulated heat range.
        HsO : Integer
            Upper boundary of sub-optimal accumulated heat range.
        HO : Integer
            Upper boundary of not-suitable accumulated heat range.

        Returns
        -------
        None.

        """
        self.LnS = int(LnS)  # Lower boundary/ not suitable
        self.LsO = int(LsO)  # Lower boundary/ sub optimal
        self.LO = int(LO)  # Lower boundary / optimal
        self.HnS = int(HnS)  # Upper boundary/ not suitable
        self.HsO = int(HsO)  # Upper boundary / sub-optimal
        self.HO = int(HO)  # Upper boundary / optimal
        self.set_Tsum_screening = True

    def setPermafrostScreening(self, permafrost_class):

        self.permafrost_class = permafrost_class  # permafrost class 2D numpy array
        self.set_Permafrost_screening = True

    def setupTypeBConstraint(self, file_path, crop_name):
        """
        Optional function initiates the type B constraint (Temperature Profile 
        Constraint) on the existing crop based on user-specified constraint rules.

        Parameters
        ----------
        file_path : xlsx
            The file path of excel sheet where the Type B constraint rules are provided.
        crop_name : String
            Unique name of crop to consider. The name must be the same provided in excel sheet.


        Returns
        -------
        None.

        """

        data = pd.read_excel(file_path)
        self.crop_name = crop_name

        self.data = data.loc[data['Crop'] == self.crop_name]

        self.setTypeBConstraint = True

        # releasing data
        del (data)

    """ The main functions of MODULE II: Crop Simulation"""

    def simulateCropCycle(self, start_doy=1, end_doy=365, step_doy=1, leap_year=False):
        """Running the crop cycle calculation/simulation

        Args:
            start_doy (int, optional): Starting Julian day for simulating period. Defaults to 1.
            end_doy (int, optional): Ending Julian day for simulating period. Defaults to 365.
            step_doy (int, optional): Spacing (in days) between 2 adjacent crop simulations. Defaults to 1.
            leap_year (bool, optional): whether or not the simulating year is a leap year. Defaults to False.

        """        

        # just a counter to keep track of progress
        count_pixel_completed = 0
        total = self.im_height * self.im_width
        # this stores final result
        self.final_yield_rainfed = np.zeros((self.im_height, self.im_width))
        self.final_yield_irrig = np.zeros((self.im_height, self.im_width))
        self.crop_calender_irr = np.zeros((self.im_height, self.im_width), dtype=int)
        self.crop_calender_rain = np.zeros((self.im_height, self.im_width), dtype=int)
        
        if not self.perennial:
            self.fc2 = np.zeros((self.im_height, self.im_width))

        
        self.fc1_rain = np.zeros((self.im_height, self.im_width))
        self.fc1_irr = np.zeros((self.im_height, self.im_width))


        for i_row in range(self.im_height):

            for i_col in range(self.im_width):

                print('\nrow_col= {}_{}'.format(i_row, i_col))

                

                # check current location (pixel) is outside of study area or not. if it's outside of study area goes to next location (pixel)
                # Those unsuitable
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        count_pixel_completed = count_pixel_completed + 1
                        print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                        continue

                # 2. Permafrost screening
                if self.set_Permafrost_screening:
                    if np.logical_or(self.permafrost_class[i_row, i_col] == 1, self.permafrost_class[i_row, i_col] == 2):
                        count_pixel_completed = count_pixel_completed + 1
                        print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                        continue

                # Thermal Climate Screening
                if self.set_tclimate_screening:
                    if self.t_climate[i_row, i_col] in self.no_t_climate:
                        count_pixel_completed = count_pixel_completed + 1
                        
                        print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                       
                        continue
                
                # Minimum temperature requirement Checking
                if np.round(np.mean(self.meanT_daily[i_row, i_col,:]), 0) < self.min_temp:
                    count_pixel_completed = count_pixel_completed + 1
                        
                    print('\rDone %: ' + str(round(count_pixel_completed /
                        total*100, 2)), end='\r')
                    continue
                # print(r'\nRow{}, Col{} '.format(i_row, i_col), end = '\n')

                count_pixel_completed = count_pixel_completed + 1       
                # this allows handing leap and non-leap year differently. This is only relevant for monthly data because this value will be used in interpolations.
                # In case of daily data, length of vector will be taken as number of days in  a year.
                if leap_year:
                    days_in_year = 366
                else:
                    days_in_year = 365

                # extract climate data for particular location. And if climate data are monthly data, they are interpolated as daily data
                if self.set_monthly:
                    obj_utilities = UtilitiesCalc.UtilitiesCalc()

                    minT_daily_point = obj_utilities.interpMonthlyToDaily(
                        self.minT_monthly[i_row, i_col, :], 1, days_in_year)
                    maxT_daily_point = obj_utilities.interpMonthlyToDaily(
                        self.maxT_monthly[i_row, i_col, :], 1, days_in_year)
                    shortRad_daily_point = obj_utilities.interpMonthlyToDaily(
                        self.shortRad_monthly[i_row, i_col, :],  1, days_in_year, no_minus_values=True)
                    wind2m_daily_point = obj_utilities.interpMonthlyToDaily(
                        self.wind2m_monthly[i_row, i_col, :],  1, days_in_year, no_minus_values=True)
                    totalPrec_daily_point = obj_utilities.interpMonthlyToDaily(
                        self.totalPrec_monthly[i_row, i_col, :],  1, days_in_year, no_minus_values=True)
                    rel_humidity_daily_point = obj_utilities.interpMonthlyToDaily(
                        self.rel_humidity_monthly[i_row, i_col, :],  1, days_in_year, no_minus_values=True)
                else:
                    minT_daily_point = self.minT_daily[i_row, i_col, :]
                    maxT_daily_point = self.maxT_daily[i_row, i_col, :]
                    shortRad_daily_point = self.shortRad_daily[i_row, i_col, :]
                    wind2m_daily_point = self.wind2m_daily[i_row, i_col, :]
                    totalPrec_daily_point = self.totalPrec_daily[i_row, i_col, :]
                    rel_humidity_daily_point = self.rel_humidity_daily[i_row, i_col, :]
                    

                # calculate ETO for full year for particular location (pixel) 7#
                obj_eto = ETOCalc.ETOCalc(
                    1, minT_daily_point.shape[0], self.latitude[i_row, i_col], self.elevation[i_row, i_col])

                # 7. New Modification
                shortRad_daily_point_MJm2day = (shortRad_daily_point*3600*24)/1000000 # convert w/m2 to MJ/m2/day (Correct)

                # 7. Minor change for validation purposes: shortRad_daily_point is replaced in shortRad_dailyy_point_MJm2day. (Sunshine hour data for KB Etocalc)
                obj_eto.setClimateData(minT_daily_point, maxT_daily_point,
                                       wind2m_daily_point, shortRad_daily_point_MJm2day, rel_humidity_daily_point)
                pet_daily_point = obj_eto.calculateETO()

                

                """No adjustment of cycle length, LAI and HI for non-perennials"""
                if not self.perennial:
                    
                    self.cycle_len_rain = self.cycle_len
                    self.LAi_rain = self.LAi
                    self.HI_rain = self.HI

                    self.cycle_len_irr = self.cycle_len
                    self.LAi_irr = self.LAi
                    self.HI_irr = self.HI
                
                else:
                    """Adjustment of cycle length, LAI and HI for Perennials"""
                    self.set_adjustment = True

                    """ Adjustment for RAINFED conditions"""

                    if self.LGP[i_row, i_col] < self.cycle_len:

                        # LGP duration will be efficient cycle length for rainfed conditions
                        # Later, we use LGP length to adjust for LAI and HI for rainfed conditions
                        self.cycle_len_rain = int(self.LGP[i_row, i_col])
                        self.adjustForPerennialCrop(
                            self.cycle_len_rain, aLAI=self.aLAI, bLAI=self.bLAI, aHI=self.aHI, bHI=self.bHI, rain_or_irr='rain')
                    else:
                        self.cycle_len_rain = self.cycle_len
                        self.LAi_rain = self.LAi
                        self.HI_rain = self.HI
                
                    """ Adjustment for IRRIGATED conditions"""

                    """Use LGPT5 for minimum temperature less than or equal to five deg Celsius"""
                    if self.min_temp <= 5:

                        if self.LGPT5[i_row, i_col] < self.cycle_len:

                            self.cycle_len_irr = int(self.LGPT5[i_row, i_col].copy())
                            self.adjustForPerennialCrop(
                                self.cycle_len_irr, aLAI=self.aLAI, bLAI=self.bLAI, aHI=self.aHI, bHI=self.bHI, rain_or_irr='irr')

                        else:
                            self.cycle_len_irr = self.cycle_len
                            self.LAi_irr = self.LAi
                            self.HI_irr = self.HI

                    """Use LGPT10 for minimum temperature greater than five deg Celsius"""

                    if self.min_temp > 5:

                        if self.LGPT10[i_row, i_col] < self.cycle_len:

                            self.cycle_len_irr = int(self.LGPT10[i_row, i_col].copy())
                            self.adjustForPerennialCrop(
                                self.cycle_len_irr, aLAI=self.aLAI, bLAI=self.bLAI, aHI=self.aHI, bHI=self.bHI, rain_or_irr='irr')

                        else:
                            self.cycle_len_irr = self.cycle_len
                            self.LAi_irr = self.LAi
                            self.HI_irr = self.HI
                

                # Empty arrays that stores yield estimations and fc1 and fc2 of all cycles per particular location (pixel)
                yield_of_all_crop_cycles_rainfed = np.empty(0, dtype= np.float16)
                yield_of_all_crop_cycles_irrig = np.empty(0, dtype= np.float16)

                fc1_rain_lst = np.empty(0, dtype= np.float16)
                fc1_irr_lst = np.empty(0, dtype= np.float16)

                fc2_lst = np.empty(0, dtype= np.float16)


                """ Calculation of each individual day's yield for rainfed and irrigated conditions"""

                for i_cycle in range(start_doy-1, end_doy, step_doy):

                    """Repeat the climate data two times and concatenate for computational convenience. If perennial, the cycle length
                    will be different for separate conditions"""
                    print('Cycle No.{}'.format(i_cycle), end = '\n')

                    minT_daily_2year = np.tile(minT_daily_point, 2)
                    maxT_daily_2year = np.tile(maxT_daily_point, 2)
                    shortRad_daily_2year = np.tile(shortRad_daily_point, 2)
                    
                    totalPrec_daily_2year = np.tile(totalPrec_daily_point, 2)
                    pet_daily_2year = np.tile(pet_daily_point, 2)

                    # print('Tiling complete')

                    

                    """ Time slicing tiled climate data with corresponding cycle lengths for rainfed and irrigated conditions"""
                    """For rainfed"""

                    # extract climate data within the season to pass in to calculation classes
                    minT_daily_season_rain = minT_daily_2year[i_cycle: i_cycle +
                                                                int(self.cycle_len_rain)-1]
                    maxT_daily_season_rain = maxT_daily_2year[i_cycle: i_cycle +
                                                                int(self.cycle_len_rain)-1]
                    shortRad_daily_season_rain = shortRad_daily_2year[
                        i_cycle: i_cycle+int(self.cycle_len_rain)-1]
                    pet_daily_season_rain = pet_daily_2year[
                        i_cycle: i_cycle+int(self.cycle_len_rain)-1]
                    totalPrec_daily_season_rain = totalPrec_daily_2year[
                        i_cycle: i_cycle+int(self.cycle_len_rain)-1]
                    
                    """For irrigated"""
                    # extract climate data within the season to pass in to calculation classes
                    minT_daily_season_irr = minT_daily_2year[i_cycle: i_cycle +
                                                                int(self.cycle_len_irr)-1]
                    maxT_daily_season_irr = maxT_daily_2year[i_cycle: i_cycle +
                                                                int(self.cycle_len_irr)-1]
                    shortRad_daily_season_irr = shortRad_daily_2year[
                        i_cycle: i_cycle+int(self.cycle_len_irr)-1]
                    pet_daily_season_irr = pet_daily_2year[
                        i_cycle: i_cycle+int(self.cycle_len_irr)-1]
                    totalPrec_daily_season_irr = totalPrec_daily_2year[
                        i_cycle: i_cycle+int(self.cycle_len_irr)-1]
                    
                    # print('Climate time slicing complete')



                    """Thermal Screening using each cycle length for rainfed and irrigated conditions"""

                    """ For the perennial, the adjusted cycle length for rainfed and irrigated conditions will be used. For the rest,
                        the user-specified cycle length will be applied"""

                    """Creating Thermal Screening object classes for perennial rainfed and irrigated conditions"""
                    obj_screening_rain = ThermalScreening.ThermalScreening()
                    obj_screening_irr = ThermalScreening.ThermalScreening()


                    obj_screening_rain.setClimateData(
                        minT_daily_season_rain, maxT_daily_season_rain)
                    obj_screening_irr.setClimateData(
                        minT_daily_season_irr, maxT_daily_season_irr)


                    if self.set_lgpt_screening:
                        obj_screening_rain.setLGPTScreening(
                            no_lgpt=self.no_lgpt, optm_lgpt=self.optm_lgpt)
                        obj_screening_irr.setLGPTScreening(
                            no_lgpt=self.no_lgpt, optm_lgpt=self.optm_lgpt)

                    # 5 Modification (SWH)
                    if self.set_Tsum_screening:
                        obj_screening_rain.setTSumScreening(
                            LnS=self.LnS, LsO=self.LsO, LO=self.LO, HnS=self.HnS, HsO=self.HsO, HO=self.HO)
                        obj_screening_irr.setTSumScreening(
                            LnS=self.LnS, LsO=self.LsO, LO=self.LO, HnS=self.HnS, HsO=self.HsO, HO=self.HO)

                    # 8 Modification
                    if self.setTypeBConstraint:

                        if self.perennial:
                            obj_screening_rain.applyTypeBConstraint(
                                data=self.data, input_temp_profile=obj_screening_rain.tprofile, perennial_flag=True)
                            obj_screening_irr.applyTypeBConstraint(
                                data=self.data, input_temp_profile=obj_screening_irr.tprofile, perennial_flag=True)
                        else:
                            obj_screening_rain.applyTypeBConstraint(
                            data=self.data, input_temp_profile=obj_screening_rain.tprofile, perennial_flag=False)
                            obj_screening_irr.applyTypeBConstraint(
                            data=self.data, input_temp_profile=obj_screening_irr.tprofile, perennial_flag=False)

                    fc1_rain = 1.
                    fc1_irr = 1.
                    # print('Original fc1_rain =', fc1_rain)
                    # print('Original fc1_irr =', fc1_irr)

                    fc1_rain = obj_screening_rain.getReductionFactor2()  # fc1 for rainfed condition
                    fc1_irr = obj_screening_irr.getReductionFactor2()  # fc1 for irrigated condition

                

                    # if not obj_screening_rain.getSuitability():
                    #     continue

                    # else:
                    #     fc1_rain = obj_screening_rain.getReductionFactor2()  # fc1 for rainfed condition

                    # if not obj_screening_irr.getSuitability():
                    #     continue
                    # else:
                    #     fc1_irr = obj_screening_irr.getReductionFactor2()  # fc1 for irrigated condition
                    
                    if fc1_rain == None or fc1_irr == None:
                        raise Exception('Fc1 not returned in Thermal Screening Calculation. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                    
                    if fc1_rain == np.nan or fc1_irr == np.nan:
                        raise Exception('Fc1 nan value returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                    
                    # Appending individual cycle's fc1 for rainfed and irrigated condition
                    # print('After fc1 irr=', fc1_irr)
                    # print('After fc1 rain=', fc1_rain)

                    # print('Thermal Screening calculation complete')
                    
                    fc1_irr_lst = np.append(fc1_irr_lst, fc1_irr)
                    fc1_rain_lst = np.append(fc1_rain_lst, fc1_rain)


                    # print('fc1_irr_lst = ', fc1_irr_lst)
                    # print('fc1_rain_lst = ', fc1_rain_lst)

                    if len(fc1_irr_lst) != i_cycle+1:
                        print('fc1_irr_lst = ', fc1_irr_lst)
                        raise Exception('Fc1 irr not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))
                    
                    elif len(fc1_rain_lst)!= i_cycle+1:
                        raise Exception('Fc1 rain not properly appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))


                    """Biomass Calculation relevant to perennials and non-perennials for IRRIGATED conditions"""
                    """IRRIGATED"""
                    obj_maxyield_irr = BioMassCalc.BioMassCalc(
                        i_cycle+1, i_cycle+1+self.cycle_len_irr-1, self.latitude[i_row, i_col])
                    obj_maxyield_irr.setClimateData(
                        minT_daily_season_irr, maxT_daily_season_irr, shortRad_daily_season_irr)
                    obj_maxyield_irr.setCropParameters(
                        self.LAi_irr, self.HI_irr, self.legume, self.adaptability)
                    obj_maxyield_irr.calculateBioMass()
                    est_yield_irrigated = obj_maxyield_irr.calculateYield()

                    # reduce thermal screening factor
                    est_yield_irrigated = est_yield_irrigated * fc1_irr

                    if est_yield_irrigated == None or est_yield_irrigated == np.nan:
                        raise Exception('Irrigated Yield not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))


                    """append current cycle yield to a list IRRIGATED"""
                    yield_of_all_crop_cycles_irrig = np.append(yield_of_all_crop_cycles_irrig, est_yield_irrigated)

                    if len(yield_of_all_crop_cycles_irrig) != i_cycle+1:
                        raise Exception('Irr Yield cycles not appended. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))

                    



                    """A separate biomass calculation RAINFED for Non-Perennials"""
                    obj_maxyield_rain = BioMassCalc.BioMassCalc(
                        i_cycle+1, i_cycle+1+self.cycle_len_rain-1, self.latitude[i_row, i_col])
                    obj_maxyield_rain.setClimateData(
                        minT_daily_season_rain, maxT_daily_season_rain, shortRad_daily_season_rain)
                    obj_maxyield_rain.setCropParameters(
                        self.LAi_rain, self.HI_rain, self.legume, self.adaptability)
                    obj_maxyield_rain.calculateBioMass()
                    est_yield_rainfed = obj_maxyield_rain.calculateYield()

                    # reduce thermal screening factor
                    est_yield_rainfed = est_yield_rainfed * fc1_rain

                    if est_yield_rainfed == None or est_yield_rainfed == np.nan:
                        raise Exception('Biomass Yield for rainfed not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))



                    """For RAINFED Perennials, fc2 will be zero (No Need for Crop Water Requirement calculation)"""
                    if self.perennial:
                        est_yield_rainfed = est_yield_rainfed * 1

                        """append current cycle yield to a list RAINFED"""
                        yield_of_all_crop_cycles_rainfed = np.append(yield_of_all_crop_cycles_rainfed, est_yield_rainfed)

                    else:
                        obj_cropwat = CropWatCalc.CropWatCalc(
                            i_cycle+1, i_cycle+1+self.cycle_len_rain-1)
                        obj_cropwat.setClimateData(
                            pet_daily_season_rain, totalPrec_daily_season_rain)
                        
                        # check Sa is a raster or single value and extract Sa value accordingly
                        if len(np.array(self.Sa).shape) == 2:
                            Sa_temp = self.Sa[i_row, i_col]
                        else:
                            Sa_temp = self.Sa
                        obj_cropwat.setCropParameters(self.d_per, self.kc, self.kc_all, self.yloss_f,
                                                      self.yloss_f_all, est_yield_rainfed, self.D1, self.D2, Sa_temp, self.pc)
                        est_yield_moisture_limited = obj_cropwat.calculateMoistureLimitedYield()

                        if est_yield_moisture_limited == None or est_yield_moisture_limited == np.nan:
                            raise Exception('Crop Water Yield not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))

                        fc2_value = obj_cropwat.getfc2factormap()

                        if fc2_value == None or fc2_value == np.nan:
                            raise Exception('fc2 value not returned. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))

                        """append current cycle yield to a list RAINFED"""
                        yield_of_all_crop_cycles_rainfed = np.append(yield_of_all_crop_cycles_rainfed, est_yield_moisture_limited)
                        fc2_lst = np.append(fc2_lst, fc2_value)

                        if len(yield_of_all_crop_cycles_rainfed) != i_cycle+1:
                            raise Exception('Rainfed yield list not properly appended')
                        elif len(fc2_lst) != i_cycle+1:
                            raise Exception('Fc2 list not appended properly. Row_{}_col_{}_Cycle_{}'.format(i_row, i_col, i_cycle))

                            # print(r'rain_yield_length = ', len(yield_of_all_crop_cycles_rainfed))
                            # print(r'fc2 = ', fc2_value)
                            # print(r'fc2 list =', len(fc2_lst))


                """Getting Maximum Attainable Yield from the list for irrigated and rainfed conditions and the Crop Calendar"""

                # get agro-climatic yield and crop calendar for IRRIGATED condition
                if np.logical_and(len(yield_of_all_crop_cycles_irrig) == len(fc1_irr_lst), len(yield_of_all_crop_cycles_irrig) == len(fc1_irr_lst)):

                    self.final_yield_irrig[i_row, i_col] = np.max(yield_of_all_crop_cycles_irrig) # Maximum attainable yield

                    i = np.where(yield_of_all_crop_cycles_irrig == np.max(yield_of_all_crop_cycles_irrig))[0][0] # index of maximum yield

                    self.crop_calender_irr[i_row, i_col] = int(i+1)*step_doy # Crop calendar for irrigated condition

                    self.fc1_irr[i_row, i_col] = fc1_irr_lst[i] # fc1 irrigated for the specific crop calendar DOY

                # get agro-climatic yield and crop calendar for RAINFED condition
                if np.logical_and(len(yield_of_all_crop_cycles_rainfed) == len(fc1_rain_lst), len(yield_of_all_crop_cycles_rainfed) == len(fc1_rain_lst)):
                    self.final_yield_rainfed[i_row, i_col] = np.max(yield_of_all_crop_cycles_rainfed) # Maximum attainable yield

                    i1 = np.where(yield_of_all_crop_cycles_rainfed == np.max(yield_of_all_crop_cycles_rainfed))[0][0] # index of maximum yield
                    
                    self.crop_calender_rain[i_row, i_col] = int(i1+1) * step_doy # Crop calendar for rainfed condition
                    
                    self.fc1_rain[i_row, i_col] = fc1_rain_lst[i1]
                    
                    if not self.perennial:
                        self.fc2[i_row, i_col] = fc2_lst[i1]


                print('\rDone %: ' + str(round(count_pixel_completed / total*100, 2)), end='\r')
        
        print('\nSimulations Completed !')

    def getEstimatedYieldRainfed(self):
        """Estimation of Maximum Yield for Rainfed scenario

        Returns:
            2D NumPy: the maximum attainable yield under the provided climate conditions, 
                      under rain-fed conditions [kg/ha]
        """        
        return self.final_yield_rainfed

    def getEstimatedYieldIrrigated(self):
        """Estimation of Maximum Yield for Irrigated scenario

        Returns:
            2D NumPy: the maximum attainable yield under the provided climate conditions, 
                      under irrigated conditions [kg/ha]
        """
        return self.final_yield_irrig

    def getOptimumCycleStartDateIrrigated(self):
        """
        Function for optimum starting date for irrigated condition.

        Returns
        -------
        TYPE: 2-D numpy array.
            Optimum starting date for irrigated condition.

        """
        return self.crop_calender_irr

    def getOptimumCycleStartDateRainfed(self):
        """
        Function for optimum starting date for rainfed condition.

        Returns
        -------
        TYPE: 2-D numpy array.
            Optimum starting date for rainfed condition.

        """
        return self.crop_calender_rain

    def getThermalReductionFactor(self):
        """
        Function for thermal reduction factor (fc1) map. For perennial crop,
        the function produces a list of fc1 maps for both conditions. Only one 
        fc1 map is produced for non-perennial crops, representing both rainfed 
        and irrigated conditions

        Returns
        -------
        TYPE: A python list of 2-D numpy arrays: [fc1 rainfed, fc1 irrigated] 
        or a 2-D numpy array.
            Thermal reduction factor map (fc1) for corresponding conditions.

        """
        return [self.fc1_rain, self.fc1_irr]

    def getMoistureReductionFactor(self):
        """
        Function for reduction factor map due to moisture deficit (fc2) for 
        rainfed condition. Only fc2 map is produced for non-perennial crops.
        
        Returns
        -------
        TYPE: 2-D numpy array
            Reduction factor due to moisture deficit (fc2).

        """

        if not self.perennial:
            return self.fc2
        else:
            print('Map is not produced because moisture deficit does not apply limitation to Perennials')

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

#----------------- End of file -------------------------#
