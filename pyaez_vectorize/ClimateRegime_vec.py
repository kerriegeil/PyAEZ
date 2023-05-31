"""
PyAEZ version 2.0.0 (November 2022)
This ClimateRegime Class read/load and calculates the agro-climatic indicators
required to run PyAEZ.  
2021: N. Lakmal Deshapriya
2022: Swun Wunna Htet, K. Boonma
"""

import numpy as np

import UtilitiesCalc_vec as UtilitiesCalc
import ETOCalc_vec as ETOCalc 
import LGPCalc_vec as LGPCalc
from collections import OrderedDict

# Initiate ClimateRegime Class instance
class ClimateRegime(object):

    def setLocationTerrainData(self, lats, elevation):
    # def setLocationTerrainData(self, lat_min, lat_max, elevation):
        """Load geographical extents and elevation data in to the Class, 
           and create a latitude map

        Args:
            lat_min (float): the minimum latitude of the AOI in decimal degrees
            lat_max (float): the maximum latitude of the AOI in decimal degrees
            elevation (2D NumPy): elevation map in metre
        """        
        self.elevation = elevation
        self.im_height = elevation.shape[0]
        self.im_width = elevation.shape[1]
        # self.latitude = UtilitiesCalc.UtilitiesCalc().generateLatitudeMap(lat_min, lat_max, self.im_height, self.im_width)
        self.latitude = UtilitiesCalc.UtilitiesCalc().generateLatitudeMap(lats, elevation.shape[1])


    def setStudyAreaMask(self, admin_mask, no_data_value):
        """Set clipping mask of the area of interest (optional)

        Args:
            admin_mask (2D NumPy/Binary): mask to extract only region of interest
            no_data_value (int): pixels with this value will be omitted during PyAEZ calculations
        """    
        self.im_mask = admin_mask
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
        # y=min_temp.y.data
        # x=min_temp.x.data
        daystart=1
        dayend=365
        # t=np.arange(daystart,dayend+1)
        # init_val=0

        # implement range limitations/corrections on input data values 
        rel_humidity[rel_humidity > 0.99] = 0.99
        rel_humidity[rel_humidity < 0.05] = 0.05
        short_rad[short_rad < 0] = 0
        wind_speed[wind_speed < 0] = 0

        # we don't need to preallocate these destination arrays
        # self.meanT_daily = np.zeros((self.im_height, self.im_width, 365))
        # self.totalPrec_daily = np.zeros((self.im_height, self.im_width, 365))
        # self.pet_daily = np.zeros((self.im_height, self.im_width, 365))
        # self.minT_daily = np.zeros((self.im_height, self.im_width, 365))
        # self.maxT_daily = np.zeros((self.im_height, self.im_width, 365))


        # Interpolate monthly to daily data
        obj_utilities = UtilitiesCalc.UtilitiesCalc()

        meanT_monthly = (min_temp+max_temp)/2.

        # eliminate these slow loops
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

        self.meanT_daily=obj_utilities.interpMonthlyToDaily(meanT_monthly,daystart,dayend)
        self.totalPrec_daily=obj_utilities.interpMonthlyToDaily(precipitation,daystart,dayend,no_minus_values=True)
        self.minT_daily=obj_utilities.interpMonthlyToDaily(min_temp,daystart,dayend)
        self.maxT_daily=obj_utilities.interpMonthlyToDaily(max_temp,daystart,dayend)
        radiation_daily=obj_utilities.interpMonthlyToDaily(short_rad,daystart,dayend,no_minus_values=True)
        wind_daily=obj_utilities.interpMonthlyToDaily(wind_speed,daystart,dayend)
        rel_humidity_daily=obj_utilities.interpMonthlyToDaily(rel_humidity,daystart,dayend,no_minus_values=True)

        # calculation of reference evapotranspiration (ETo)
        obj_eto = ETOCalc.ETOCalc(daystart, dayend, self.latitude, self.elevation)
        shortrad_daily_MJm2day = (radiation_daily*3600*24)/1000000 # convert w/m2 to MJ/m2/day
        obj_eto.setClimateData(self.minT_daily, self.maxT_daily, wind_daily, shortrad_daily_MJm2day, rel_humidity_daily)
        self.pet_daily= obj_eto.calculateETO()        
                
        # Sea-level adjusted mean temperature
        # self.meanT_daily_sealevel = self.meanT_daily + np.tile(np.reshape(self.elevation/100*0.55, (self.im_height,self.im_width,1)), (1,1,365))
        self.meanT_daily_sealevel = self.meanT_daily + (self.elevation/100*0.55) # automatic broadcasting
        
        
        # P over PET ratio(to eliminate nan in the result, nan is replaced with zero)
        # self.P_by_PET_daily = np.nan_to_num(self.totalPrec_daily / self.pet_daily)
        self.P_by_PET_daily=(self.totalPrec_daily/self.pet_daily).fillna(0)

        # add interp_daily_temp

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
        # y=min_temp.y.data
        # x=min_temp.x.data
        daystart=1
        dayend=365
        # t=np.arange(daystart,dayend+1)
        # init_val=0

        # implement range limitations/corrections on input data values 
        rel_humidity[rel_humidity > 0.99] = 0.99
        rel_humidity[rel_humidity < 0.05] = 0.05
        short_rad[short_rad < 0] = 0
        wind_speed[wind_speed < 0] = 0
        
        # # Interpolate monthly to daily data
        # obj_utilities = UtilitiesCalc.UtilitiesCalc()

        self.meanT_daily = (min_temp+max_temp)/2.
        self.totalPrec_daily = precipitation
        self.maxT_daily = max_temp

        # calculation of reference evapotranspiration (ETo)
        obj_eto = ETOCalc.ETOCalc(daystart, dayend, self.latitude, self.elevation)
        shortrad_daily_MJm2day = (short_rad*3600.*24.)/1000000. # convert w/m2 to MJ/m2/day
        obj_eto.setClimateData(min_temp, max_temp, wind_speed, shortrad_daily_MJm2day, rel_humidity)
        self.pet_daily= obj_eto.calculateETO() 

        # sea level temperature
        self.meanT_daily_sealevel = self.meanT_daily + np.expand_dims(self.elevation/100*0.55,axis=2) # automatic broadcasting         
        
        # P over PET ratio (to eliminate nan in the result, nan is replaced with zero)
        self.P_by_PET_daily = np.nan_to_num(self.totalPrec_daily / self.pet_daily)

        # smoothed mean T
        # Adding interpolation to the dataset
        # 5th degree spline fit to smooth in time
        days = np.arange(1,366) # x values
        # replace any nan with zero
        mask3D = np.tile(self.im_mask[:,:,np.newaxis], (1,1,days.shape[0]))
        data=np.where(mask3D==0,0,self.meanT_daily)
        data2D=data.transpose(2,0,1).reshape(days.shape[0],-1) # every column is a set of y values
        # do the fitting
        quad_spl=np.polynomial.polynomial.polyfit(days,data2D,deg=5)
        interp_daily_temp=np.polynomial.polynomial.polyval(days,quad_spl)
        #reshape
        interp_daily_temp=interp_daily_temp.reshape(mask3D.shape[0],mask3D.shape[1],-1)
        self.interp_daily_temp=interp_daily_temp        


    def getThermalClimate(self):
        """Classification of rainfall and temperature seasonality into thermal climate classes

        Returns:
            2D NumPy: Thermal Climate classification
        """        
        # Note that currently, this thermal climate is designed only for the northern hemisphere, southern hemisphere is not implemented yet.

        # converting daily to monthly
        obj_utilities = UtilitiesCalc.UtilitiesCalc()
        meanT_monthly_sealevel = obj_utilities.averageDailyToMonthly(self.meanT_daily_sealevel)
        meanT_monthly = obj_utilities.averageDailyToMonthly(self.meanT_daily)
        P_by_PET_monthly = obj_utilities.averageDailyToMonthly(self.P_by_PET_daily)

        # things we need to determine the classes
        # compute them here for readability below
        P_by_PET_AMJJAS=P_by_PET_monthly[:,:,3:9].mean(axis=2) # Apr-Sep mean
        JFMSON=[0,1,2,9,10,11]
        P_by_PET_ONDJFM=P_by_PET_monthly[:,:,JFMSON].mean(axis=2) # Oct-Mar mean
        min_sealev_meanT=meanT_monthly_sealevel.min(axis=2)
        range_meanT=meanT_monthly.max(axis=2) - meanT_monthly.min(axis=2)
        meanT=meanT_monthly.mean(axis=2)
        nmo_ge_10C=(meanT_monthly_sealevel >= 10).sum(axis=2)
        prsum=self.totalPrec_daily.sum(axis=2)

        # determine the classes
        thermal_climate = np.empty((meanT_monthly.shape[0],meanT_monthly.shape[1]))
        thermal_climate[:] = np.nan

        # Tropics
        # Tropical lowland
        thermal_climate=np.where((min_sealev_meanT>18) & (range_meanT<15) & (meanT>20),1,thermal_climate)
        # Tropical highland
        thermal_climate=np.where((min_sealev_meanT>18) & (range_meanT<15) & (meanT<=20) & ~np.isfinite(thermal_climate),2,thermal_climate)
        # SubTropic
        # Subtropics Low Rainfall
        thermal_climate=np.where((min_sealev_meanT>5) & (nmo_ge_10C>=8) & (prsum<250) & ~np.isfinite(thermal_climate),3,thermal_climate)
        # Subtropics Summer Rainfall
        thermal_climate=np.where((min_sealev_meanT>5) & (nmo_ge_10C>=8) & (self.latitude>=0) & (P_by_PET_AMJJAS>=P_by_PET_ONDJFM) & ~np.isfinite(thermal_climate),4,thermal_climate)
        thermal_climate=np.where((min_sealev_meanT>5) & (nmo_ge_10C>=8) & (self.latitude<0) & (P_by_PET_AMJJAS<=P_by_PET_ONDJFM) & ~np.isfinite(thermal_climate),4,thermal_climate)
        # Subtropics Winter Rainfall
        thermal_climate=np.where((min_sealev_meanT>5) & (nmo_ge_10C>=8) & (self.latitude>=0) & (P_by_PET_AMJJAS<=P_by_PET_ONDJFM) & ~np.isfinite(thermal_climate),5,thermal_climate)
        thermal_climate=np.where((min_sealev_meanT>5) & (nmo_ge_10C>=8) & (self.latitude<0) & (P_by_PET_AMJJAS>=P_by_PET_ONDJFM) & ~np.isfinite(thermal_climate),5,thermal_climate)
        # Temperate
        # Oceanic Temperate
        thermal_climate=np.where((nmo_ge_10C>=4) & (range_meanT<=20) & ~np.isfinite(thermal_climate),6,thermal_climate)
        # Sub-Continental Temperate
        thermal_climate=np.where((nmo_ge_10C>=4) & (range_meanT<=35) & ~np.isfinite(thermal_climate),7,thermal_climate)
        # Continental Temperate
        thermal_climate=np.where((nmo_ge_10C>=4) & (range_meanT>35) & ~np.isfinite(thermal_climate),8,thermal_climate)
        # Boreal
        # Oceanic Boreal
        thermal_climate=np.where((nmo_ge_10C>=1) & (range_meanT<=20) & ~np.isfinite(thermal_climate),9,thermal_climate)
        # Sub-Continental Boreal
        thermal_climate=np.where((nmo_ge_10C>=1) & (range_meanT<=35) & ~np.isfinite(thermal_climate),10,thermal_climate)
        # Continental Boreal
        thermal_climate=np.where((nmo_ge_10C>=1) & (range_meanT>35) & ~np.isfinite(thermal_climate),11,thermal_climate)
        # Arctic
        thermal_climate=np.where((self.im_mask==1) & ~np.isfinite(thermal_climate),12,thermal_climate)
        thermal_climate=np.where(self.im_mask==0,0,thermal_climate).astype('int8')        
        
        if self.set_mask:
            return np.ma.masked_where(self.im_mask == 0, thermal_climate)
        else:
            return thermal_climate
    

    def getThermalZone(self):
        """The thermal zone is classified based on actual temperature which reflects 
        on the temperature regimes of major thermal climates

        Returns:
            2D NumPy: Thermal Zones classification
        """  
        # converting daily to monthly
        obj_utilities = UtilitiesCalc.UtilitiesCalc()
        meanT_monthly_sealevel = obj_utilities.averageDailyToMonthly(self.meanT_daily_sealevel)
        meanT_monthly = obj_utilities.averageDailyToMonthly(self.meanT_daily)

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


        # determine the classes
        thermal_zone = np.empty((meanT_monthly.shape[0],meanT_monthly.shape[1]))
        thermal_zone[:] = np.nan

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
        thermal_zone=np.where((self.im_mask==1) & ~np.isfinite(thermal_zone),12,thermal_zone)
        thermal_zone=np.where(self.im_mask==0,0,thermal_zone).astype('int8') 
    
        if self.set_mask:
            return np.ma.masked_where(self.im_mask == 0, thermal_zone)
        else:
            return thermal_zone

    def getThermalLGP0(self):
        """Calculate Thermal Length of Growing Period (LGPt) with 
        temperature threshold of 0 degree Celcius

        Returns:
            2D numpy: The accumulated number of days with daily mean 
                      temperature is above 0 degree Celcius
        """     
        # added this to set*ClimateData
        # # # Adding interpolation to the dataset
        # # 5th degree spline fit to smooth in time
        # days = np.arange(1,366) # x values
        # # replace any nan with zero
        # mask3D = np.tile(self.im_mask[:,:,np.newaxis], (1,1,days.shape[0]))
        # data=np.where(mask3D==0,0,self.meanT_daily)
        # data2D=data.transpose(2,0,1).reshape(days.shape[0],-1) # every column is a set of y values
        # # do the fitting
        # quad_spl=np.polynomial.polynomial.polyfit(days,data2D,deg=5)
        # interp_daily_temp=np.polynomial.polynomial.polyval(days,quad_spl)
        # #reshape
        # interp_daily_temp=interp_daily_temp.reshape(mask3D.shape[0],mask3D.shape[1],-1)

        # count number of days above threshold
        lgpt0 = np.sum(self.interp_daily_temp>=0, axis=2)
        if self.set_mask:
            lgpt0 = np.ma.masked_where(self.im_mask == 0, lgpt0)
        
        self.lgpt0=lgpt0.copy()
        return lgpt0           


    def getThermalLGP5(self):
        """Calculate Thermal Length of Growing Period (LGPt) with 
        temperature threshold of 5 degree Celcius

        Returns:
            2D numpy: The accumulated number of days with daily mean 
                      temperature is above 5 degree Celcius
        """     
        # added this to set*ClimateData
        # # # Adding interpolation to the dataset
        # # 5th degree spline fit to smooth in time
        # days = np.arange(1,366) # x values
        # # replace any nan with zero
        # mask3D = np.tile(self.im_mask[:,:,np.newaxis], (1,1,days.shape[0]))
        # data=np.where(mask3D==0,0,self.meanT_daily)
        # data2D=data.transpose(2,0,1).reshape(days.shape[0],-1) # every column is a set of y values
        # # do the fitting
        # quad_spl=np.polynomial.polynomial.polyfit(days,data2D,deg=5)
        # interp_daily_temp=np.polynomial.polynomial.polyval(days,quad_spl)
        # #reshape
        # interp_daily_temp=interp_daily_temp.reshape(mask3D.shape[0],mask3D.shape[1],-1)

        # count number of days above threshold
        lgpt5 = np.sum(self.interp_daily_temp>=5, axis=2)
        if self.set_mask:
            lgpt5 = np.ma.masked_where(self.im_mask == 0, lgpt5)
        
        self.lgpt5=lgpt5.copy()
        return lgpt5   

    def getThermalLGP10(self):
        """Calculate Thermal Length of Growing Period (LGPt) with
        temperature threshold of 10 degree Celcius

        Returns:
            2D numpy: The accumulated number of days with daily mean
                      temperature is above 10 degree Celcius
        """
        # added this to set*ClimateData
        # # # Adding interpolation to the dataset
        # # 5th degree spline fit to smooth in time
        # days = np.arange(1,366) # x values
        # # replace any nan with zero
        # mask3D = np.tile(self.im_mask[:,:,np.newaxis], (1,1,days.shape[0]))
        # data=np.where(mask3D==0,0,self.meanT_daily)
        # data2D=data.transpose(2,0,1).reshape(days.shape[0],-1) # every column is a set of y values
        # # do the fitting
        # quad_spl=np.polynomial.polynomial.polyfit(days,data2D,deg=5)
        # interp_daily_temp=np.polynomial.polynomial.polyval(days,quad_spl)
        # #reshape
        # interp_daily_temp=interp_daily_temp.reshape(mask3D.shape[0],mask3D.shape[1],-1)

        # count number of days above threshold
        lgpt10 = np.sum(self.interp_daily_temp>=10, axis=2)
        if self.set_mask:
            lgpt10 = np.ma.masked_where(self.im_mask == 0, lgpt10)
        
        self.lgpt10=lgpt10.copy()
        return lgpt10   

    def getTemperatureSum0(self):
        """Calculate temperature summation at temperature threshold 
        of 0 degree Celcius

        Returns:
            2D numpy: Accumulative daily average temperature (Ta) for days
                      when Ta is above the thresholds of 0 degree Celcius
        """

        tempT = self.meanT_daily.copy()
        tempT[tempT<0] = 0
        tsum0 = np.round(np.sum(tempT, axis=2), decimals = 0) 
        # masking
        if self.set_mask:
            tsum0 = np.ma.masked_where(self.im_mask == 0, tsum0)
        return tsum0

    def getTemperatureSum5(self):
        """Calculate temperature summation at temperature threshold 
        of 5 degree Celcius

        Returns:
            2D numpy: Accumulative daily average temperature (Ta) for days
                      when Ta is above the thresholds of 5 degree Celcius
        """    

        tempT = self.meanT_daily.copy()
        tempT[tempT<5] = 0
        tsum5 = np.round(np.sum(tempT, axis=2), decimals = 0) 
        # masking
        if self.set_mask: 
            tsum5 = np.ma.masked_where(self.im_mask == 0, tsum5)
        return tsum5
        

    def getTemperatureSum10(self):
        """Calculate temperature summation at temperature threshold 
        of 10 degree Celcius

        Returns:
            2D numpy: Accumulative daily average temperature (Ta) for days
                      when Ta is above the thresholds of 10 degree Celcius
        """

        tempT = self.meanT_daily.copy()
        tempT[tempT<10] = 0
        tsum10 = np.round(np.sum(tempT, axis=2), decimals = 0) 
        # masking
        if self.set_mask: 
            tsum10 = np.ma.masked_where(self.im_mask == 0, tsum10)
        return tsum10

    def getTemperatureProfile(self):
        """Classification of temperature ranges for temperature profile

        Returns:
            2D NumPy: 18 2D arrays [A1-A9, B1-B9] correspond to each Temperature Profile class [days]
        """    
        # list of variable names to compute and output
        var_names = ['A1','A2','A3','A4','A5','A6','A7','A8','A9', \
                    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']
        var_dict=OrderedDict.fromkeys(var_names)

        # added this to set*ClimateData
        # # # Adding interpolation to the dataset
        # # 5th degree spline fit to smooth in time
        # days = np.arange(1,366) # x values
        # # replace any nan with zero
        # mask3D = np.tile(self.im_mask[:,:,np.newaxis], (1,1,days.shape[0]))
        # data=np.where(mask3D==0,0,self.meanT_daily)
        # data2D=data.transpose(2,0,1).reshape(days.shape[0],-1) # every column is a set of y values
        # # do the fitting
        # quad_spl=np.polynomial.polynomial.polyfit(days,data2D,deg=5)
        # interp_daily_temp=np.polynomial.polynomial.polyval(days,quad_spl)
        # #reshape
        # interp_daily_temp=interp_daily_temp.reshape(mask3D.shape[0],mask3D.shape[1],-1)
        
        # we will use the interpolated temperature time series to decide and count
        meanT_first=self.interp_daily_temp
        meanT_diff=np.diff(self.interp_daily_temp,n=1,axis=2,append=self.interp_daily_temp[:,:,0:1]) 

        var_dict['A9'] = np.sum( np.logical_and(meanT_diff>0, meanT_first<-5), axis=2 )
        var_dict['A8'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=-5, meanT_first<0)), axis=2 )
        var_dict['A7'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=0, meanT_first<5)), axis=2 )
        var_dict['A6'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=5, meanT_first<10)), axis=2 )
        var_dict['A5'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=10, meanT_first<15)), axis=2 )
        var_dict['A4'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=15, meanT_first<20)), axis=2 )
        var_dict['A3'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=20, meanT_first<25)), axis=2 )
        var_dict['A2'] = np.sum( np.logical_and(meanT_diff>0, np.logical_and(meanT_first>=25, meanT_first<30)), axis=2 )
        var_dict['A1'] = np.sum( np.logical_and(meanT_diff>0, meanT_first>=30), axis=2 )

        var_dict['B9'] = np.sum( np.logical_and(meanT_diff<0, meanT_first<-5), axis=2 )
        var_dict['B8'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=-5, meanT_first<0)), axis=2 )
        var_dict['B7'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=0, meanT_first<5)), axis=2 )
        var_dict['B6'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=5, meanT_first<10)), axis=2 )
        var_dict['B5'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=10, meanT_first<15)), axis=2 )
        var_dict['B4'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=15, meanT_first<20)), axis=2 )
        var_dict['B3'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=20, meanT_first<25)), axis=2 )
        var_dict['B2'] = np.sum( np.logical_and(meanT_diff<0, np.logical_and(meanT_first>=25, meanT_first<30)), axis=2 )
        var_dict['B1'] = np.sum( np.logical_and(meanT_diff<0, meanT_first>=30), axis=2 )
        
        # apply the mask
        if self.set_mask:
            for var in var_names:
                var_dict[var]=np.ma.masked_where(self.im_mask == 0, var_dict[var])

        # assemble the list of return variables
        data_out=[]
        for var in var_names:
            data_out.append(var_dict[var].astype('float32'))
        
        return data_out      


    def getLGP(self, Sa=100., D=1.):
        """Calculate length of growing period (LGP)

        Args:
            Sa (float, optional): Available soil moisture holding capacity [mm/m]. Defaults to 100..
            D (float, optional): Rooting depth. Defaults to 1..

        Returns:
           2D NumPy: Length of Growing Period
        """  
        #============================
        kc_list = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
        #============================
        Txsnm = 0.  # Txsnm - snow melt temperature threshold
        Fsnm = 5.5  # Fsnm - snow melting coefficient
        Sb_old = np.zeros(self.im_mask.shape)
        Wb_old = np.zeros(self.im_mask.shape)
        #============================
        Tx365 = self.maxT_daily.copy()
        Ta365 = self.meanT_daily.copy()
        Pcp365 = self.totalPrec_daily.copy()
        self.Eto365 = self.pet_daily.copy()  # Eto

        self.Etm365=np.empty(self.Eto365.shape)
        self.Eta365=np.empty(self.Eto365.shape)
        self.Wb365=np.empty(self.Eto365.shape)
        self.Wx365=np.empty(self.Eto365.shape)
        self.Sb365=np.empty(self.Eto365.shape)
        self.kc365=np.empty(self.Eto365.shape)

        self.Etm365[:],self.Eta365[:],self.Wb365[:],self.Wx365[:],self.Sb365[:],self.kc365[:]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        #============================
        lgpt5 = self.lgpt5       
        totalPrec_monthly = UtilitiesCalc.UtilitiesCalc().averageDailyToMonthly(self.totalPrec_daily)
        self.meanT_daily_new, istart0, istart1 = LGPCalc.rainPeak(
            totalPrec_monthly, Ta365, lgpt5)
        p = LGPCalc.psh(np.zeros(self.Eto365.shape), self.Eto365)            

        for doy in range(0, 365):
            Eta_new, Etm_new, Wb_new, Wx_new, Sb_new, kc_new = LGPCalc.EtaCalc(
                                    self.im_mask,
                                    np.float64(Tx365[:,:,doy]), 
                                    np.float64(Ta365[:,:,doy]),
                                    np.float64(Pcp365[:,:,doy]), 
                                    Txsnm, 
                                    Fsnm, 
                                    np.float64(self.Eto365[:,:,doy]),
                                    Wb_old, 
                                    Sb_old, 
                                    doy, 
                                    istart0, 
                                    istart1,
                                    Sa, 
                                    D, 
                                    p[:,:,doy], 
                                    kc_list, 
                                    self.lgpt5)

            self.Etm365[:,:,doy]=Etm_new
            self.Eta365[:,:,doy]=Eta_new
            self.Wb365[:,:,doy]=Wb_new
            self.Wx365[:,:,doy]=Wx_new
            self.Sb365[:,:,doy]=Sb_new
            self.kc365[:,:,doy]=kc_new

            Wb_old=Wb_new
            Sb_old=Sb_new

        self.Eta365=np.where(self.Eta365<0,0,self.Eta365)  

        Etm365X = np.append(self.Etm365, self.Etm365[:,:,0:30],axis=2)
        Eta365X = np.append(self.Eta365, self.Eta365[:,:,0:30],axis=2)

        # eliminate call to LGPCalc.islgpt
        islgp=np.where(self.meanT_daily>=5,1,0)
        
        xx = LGPCalc.val10day(Eta365X)
        yy = LGPCalc.val10day(Etm365X)
    
        with np.errstate(divide='ignore', invalid='ignore'):
            lgp_whole = xx[:,:,:365]/yy[:,:,:365]

        lgp_tot=np.where((islgp==1)&(lgp_whole>=0.4),1,0).sum(axis=2)
        lgp_tot=np.where(self.im_mask==1,lgp_tot,np.nan)    
        
        if self.set_mask:
            return np.ma.masked_where(self.im_mask==0, lgp_tot)
        else:
            return lgp_tot
  
    def getLGPClassified(self, lgp): # Original PyAEZ source code
        """This function calculates the classification of moisture regime using LGP.

        Args:
            lgp (2D NumPy): Length of Growing Period

        Returns:
            2D NumPy: Classified Length of Growing Period
        """        
        # 

        lgp_class = np.zeros(lgp.shape)

        lgp_class[lgp>=365] = 7 # Per-humid
        lgp_class[np.logical_and(lgp>=270, lgp<365)] = 6 # Humid
        lgp_class[np.logical_and(lgp>=180, lgp<270)] = 5 # Sub-humid
        lgp_class[np.logical_and(lgp>=120, lgp<180)] = 4 # Moist semi-arid
        lgp_class[np.logical_and(lgp>=60, lgp<120)] = 3 # Dry semi-arid
        lgp_class[np.logical_and(lgp>0, lgp<60)] = 2 # Arid
        lgp_class[lgp<=0] = 1 # Hyper-arid

        if self.set_mask:
            return np.ma.masked_where(self.im_mask == 0, lgp_class)
        else:
            return lgp_class
        
        
    def getLGPEquivalent(self): 
        """Calculate the Equivalent LGP 

        Returns:
            2D NumPy: LGP Equivalent 
        """        
        moisture_index = np.sum(self.totalPrec_daily, axis=2)/np.sum(self.pet_daily, axis=2)
        # moisture_index = np.mean(self.totalPrec_daily/self.pet_daily, axis = 2)

        lgp_equv = 14.0 + 293.66*moisture_index - 61.25*moisture_index*moisture_index
        lgp_equv[ moisture_index > 2.4 ] = 366

        if self.set_mask:
            return np.ma.masked_where(self.im_mask == 0, lgp_equv)
        else:
            return lgp_equv


    
    def getMultiCroppingZones(self, t_climate, lgp, lgp_t5, lgp_t10, ts_t0, ts_t10):
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
        
        # Definition of multi-cropping classes (Reference: GAEZ v4 Model Documentation)
        # 1 => A Zone (Zone of no cropping, too cold or too dry for rainfed crops)
        # 2 => B Zone (Zone of single cropping)
        # 3 => C Zone (Zone of limited double cropping, relay cropping; single wetland rice may be possible)
        # 4 => D Zone (Zone of double cropping; sequential double cropping including wetland rice is not possible)
        # 5 => E Zone (Zone of double cropping with rice; sequential double cropping with one wetland rice crop is possible)
        # 6 => F Zone (Zone of double rice cropping or limited triple cropping; may partly involve relay cropping; a third crop is not possible in case of two wetland rice crops)
        # 7 => G Zone (Zone of triple cropping; sequential cropping of three short-cycle crops; two wetland rice crops are possible)
        # 8 => H Zone (Zone of triple rice cropping; sequential cropping of three wetland rice crops is possible)
        
        # defining the constant arrays for rainfed and irrigated conditions, all pixel values start with 1
        multi_crop_rain = np.ones((self.im_height, self.im_width)) # all values started with Zone A
        multi_crop_irr = np.ones((self.im_height, self.im_width)) # all vauels starts with Zone A

        # added this to set*ClimateData
        # # 5th degree spline fit to smooth in time
        # days = np.arange(1,366) # x values
        # # replace any nan with zero
        # mask3D = np.tile(self.im_mask[:,:,np.newaxis], (1,1,days.shape[0]))
        # data=np.where(mask3D==0,0,self.meanT_daily)
        # data2D=data.transpose(2,0,1).reshape(days.shape[0],-1) # every column is a set of y values
        # # do the fitting
        # quad_spl=np.polynomial.polynomial.polyfit(days,data2D,deg=5)
        # interp_daily_temp=np.polynomial.polynomial.polyval(days,quad_spl)
        # #reshape
        # interp_daily_temp=interp_daily_temp.reshape(mask3D.shape[0],mask3D.shape[1],-1)

        # cumulative T at different thresholds
        interp_meanT_veg_T5=np.where(self.interp_daily_temp>=5.,self.interp_daily_temp,np.nan)
        interp_meanT_veg_T10=np.where(self.interp_daily_temp>=10.,self.interp_daily_temp,np.nan)
        ts_g_t5=np.nansum(interp_meanT_veg_T5,axis=2)
        ts_g_t10=np.nansum(interp_meanT_veg_T10,axis=2)

        """Rainfed conditions"""
        """Lowland tropic climate"""
    
        zone_B = np.all([t_climate ==1, lgp>=45, lgp_t5>=120, lgp_t10>=90, ts_t0>=1600, ts_t10>=1200], axis=0)
        
        # three criteria for zone C conditions
        zone_C1 = np.all([t_climate ==1, lgp>=220, lgp_t5>=220, lgp_t10>=120, ts_t0>=5500, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        zone_C2 = np.all([t_climate ==1, lgp>=200, lgp_t5>=200, lgp_t10>=120, ts_t0>=6400, ts_g_t5>=3200, ts_g_t10>=2700], axis=0) # lgp_t5 is 200 instead of 210
        zone_C3 = np.all([t_climate ==1, lgp>=180, lgp_t5>=200, lgp_t10>=120, ts_t0>=7200, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        
        zone_C = np.logical_or.reduce([zone_C1==True, zone_C2==True, zone_C3==True], axis = 0)
        
        # three criteria for zone D conditions
        zone_D1 = np.all([t_climate ==1, lgp>=270, lgp_t5>=270, lgp_t10>=165, ts_t0>=5500, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_D2 = np.all([t_climate ==1, lgp>=240, lgp_t5>=240, lgp_t10>=165, ts_t0>=6400, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_D3 = np.all([t_climate ==1, lgp>=210, lgp_t5>=240, lgp_t10>=165, ts_t0>=7200, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        
        zone_D = np.logical_or.reduce([zone_D1==True, zone_D2==True, zone_D3==True], axis = 0)
        
        zone_F = np.all([t_climate ==1, lgp>=300, lgp_t5>=300, lgp_t10>=240, ts_t0>=7200, ts_g_t5>=5100, ts_g_t10>=4800], axis=0) # no criteria for ts_t10 in GAEZ
        zone_H = np.all([t_climate ==1, lgp>=360, lgp_t5>=360, lgp_t10>=360, ts_t0>=7200, ts_t10>=7000], axis=0) # lgp_t10 changed to 360
        
        
        """Other thermal climates"""
        zone_B_other = np.all([t_climate!= 1, lgp>=45, lgp_t5>=120, lgp_t10>=90, ts_t0>=1600, ts_t10>=1200], axis=0)
        zone_C_other = np.all([t_climate!= 1, lgp>=180, lgp_t5>=200, lgp_t10>=120, ts_t0>=3600, ts_t10>=3000, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        zone_D_other = np.all([t_climate!= 1, lgp>=210, lgp_t5>=240, lgp_t10>=165, ts_t0>=4500, ts_t10>=3600, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_E_other = np.all([t_climate!= 1, lgp>=240, lgp_t5>=270, lgp_t10>=180, ts_t0>=4800, ts_t10>=4500, ts_g_t5>=4300, ts_g_t10>=4000], axis=0)
        zone_F_other = np.all([t_climate!= 1, lgp>=300, lgp_t5>=300, lgp_t10>=240, ts_t0>=5400, ts_t10>=5100, ts_g_t5>=5100, ts_g_t10>=4800], axis=0)
        zone_G_other = np.all([t_climate!= 1, lgp>=330, lgp_t5>=330, lgp_t10>=270, ts_t0>=5700, ts_t10>=5500], axis=0)
        zone_H_other = np.all([t_climate!= 1, lgp>=360, lgp_t5>=360, lgp_t10>=330, ts_t0>=7200, ts_t10>=7000], axis=0)
        

        multi_crop_rain[np.logical_or.reduce([zone_B == True, zone_B_other == True], axis = 0)]= 2
        multi_crop_rain[np.logical_or.reduce([zone_C == True, zone_C_other == True], axis = 0)]= 3
        multi_crop_rain[np.logical_or.reduce([zone_D == True, zone_D_other == True], axis = 0)]= 4
        multi_crop_rain[zone_E_other == True]= 5
        multi_crop_rain[np.logical_or.reduce([zone_F == True, zone_F_other == True], axis = 0)]= 6
        multi_crop_rain[np.logical_or.reduce([zone_H == True, zone_H_other == True], axis = 0)]= 8
        
        ###########################################################################################################################
        """Irrigated conditions"""
        """Lowland tropic climate"""
        zone_B = np.all([t_climate ==1, lgp_t5>=120, lgp_t10>=90, ts_t0>=1600, ts_t10>=1200], axis=0)
        
        # three criteria for zone C conditions
        zone_C1 = np.all([t_climate ==1, lgp_t5>=220, lgp_t10>=120, ts_t0>=5500, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        zone_C2 = np.all([t_climate ==1, lgp_t5>=200, lgp_t10>=120, ts_t0>=6400, ts_g_t5>=3200, ts_g_t10>=2700], axis=0) # lgp_t5 is 200 instead of 210
        zone_C3 = np.all([t_climate ==1, lgp_t5>=200, lgp_t10>=120, ts_t0>=7200, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        
        zone_C = np.logical_or.reduce([zone_C1==True, zone_C2==True, zone_C3==True], axis = 0)
        
        # three criteria for zone D conditions
        zone_D1 = np.all([t_climate ==1, lgp_t5>=270, lgp_t10>=165, ts_t0>=5500, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_D2 = np.all([t_climate ==1, lgp_t5>=240, lgp_t10>=165, ts_t0>=6400, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_D3 = np.all([t_climate ==1, lgp_t5>=240, lgp_t10>=165, ts_t0>=7200, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        
        zone_D = np.logical_or.reduce([zone_D1==True, zone_D2==True, zone_D3==True], axis = 0)
        
        zone_F = np.all([t_climate ==1, lgp_t5>=300, lgp_t10>=240, ts_t0>=7200, ts_g_t5>=5100, ts_g_t10>=4800], axis=0) # no criteria for ts_t10 in GAEZ
        zone_H = np.all([t_climate ==1, lgp_t5>=360, lgp_t10>=360, ts_t0>=7200, ts_t10>=7000], axis=0) # lgp_t10 changed to 360
        
        
        """Other thermal climates"""
        zone_B_other = np.all([t_climate!= 1, lgp_t5>=120, lgp_t10>=90, ts_t0>=1600, ts_t10>=1200], axis=0)
        zone_C_other = np.all([t_climate!= 1, lgp_t5>=200, lgp_t10>=120, ts_t0>=3600, ts_t10>=3000, ts_g_t5>=3200, ts_g_t10>=2700], axis=0)
        zone_D_other = np.all([t_climate!= 1, lgp_t5>=240, lgp_t10>=165, ts_t0>=4500, ts_t10>=3600, ts_g_t5>=4000, ts_g_t10>=3200], axis=0)
        zone_E_other = np.all([t_climate!= 1, lgp_t5>=270, lgp_t10>=180, ts_t0>=4800, ts_t10>=4500, ts_g_t5>=4300, ts_g_t10>=4000], axis=0)
        zone_F_other = np.all([t_climate!= 1, lgp_t5>=300, lgp_t10>=240, ts_t0>=5400, ts_t10>=5100, ts_g_t5>=5100, ts_g_t10>=4800], axis=0)
        zone_G_other = np.all([t_climate!= 1, lgp_t5>=330, lgp_t10>=270, ts_t0>=5700, ts_t10>=5500], axis=0)
        zone_H_other = np.all([t_climate!= 1, lgp_t5>=360, lgp_t10>=330, ts_t0>=7200, ts_t10>=7000], axis=0)
        
        
        multi_crop_irr[np.logical_or.reduce([zone_B == True, zone_B_other == True], axis = 0)]= 2
        multi_crop_irr[np.logical_or.reduce([zone_C == True, zone_C_other == True], axis = 0)]= 3
        multi_crop_irr[np.logical_or.reduce([zone_D == True, zone_D_other == True], axis = 0)]= 4
        multi_crop_irr[zone_E_other == True]= 5
        multi_crop_irr[np.logical_or.reduce([zone_F == True, zone_F_other == True], axis = 0)]= 6
        multi_crop_irr[np.logical_or.reduce([zone_F == True, zone_F_other == True], axis = 0)]= 6
        multi_crop_irr[np.logical_or.reduce([zone_H == True, zone_H_other == True], axis = 0)]= 8
        
        if self.set_mask:
            return [np.ma.masked_where(self.im_mask == 0, multi_crop_rain), np.ma.masked_where(self.im_mask == 0, multi_crop_irr)]
        else:
            return [multi_crop_rain, multi_crop_irr]
    
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
        obj_utilities = UtilitiesCalc.UtilitiesCalc()
        max_monthly_meanT = obj_utilities.averageDailyToMonthly(self.meanT_daily).max(axis=2)
        annual_Tmean=np.mean(self.meanT_daily,axis=2)

        # thermal zone class definitions for fallow requirement
        tzonefallow=np.zeros(self.im_mask.shape)
        # Checking tropics thermal zone
        # Class 1: tropics, mean annual T > 25 deg C
        tzonefallow=np.where(((tzone==1)|(tzone==2))&(annual_Tmean>25),1,tzonefallow)
        # Class 2: tropics, mean annual T 20-25 deg C
        tzonefallow=np.where(((tzone==1)|(tzone==2))&(tzonefallow==0)&(annual_Tmean>20),2,tzonefallow)
        # Class 3: tropics, mean annual T 15-20 deg C
        tzonefallow=np.where(((tzone==1)|(tzone==2))&(tzonefallow==0)&(annual_Tmean>15),3,tzonefallow)
        # Class 4: tropics, mean annual T < 15 deg C
        tzonefallow=np.where(((tzone==1)|(tzone==2))&(tzonefallow==0)&(annual_Tmean<=15),4,tzonefallow)
        # Checking the non-tropical zones
        # Class 5: mean T of the warmest month > 20 deg C
        tzonefallow=np.where((tzone!=1)&(tzone!=2)&(tzonefallow==0)&(max_monthly_meanT>20),5,tzonefallow)
        tzonefallow=np.where((tzone!=1)&(tzone!=2)&(tzonefallow==0)&(max_monthly_meanT<=20),6,tzonefallow)        
        
        if self.set_mask:
            return np.ma.masked_where(self.im_mask == 0, tzonefallow)
        else:
            return tzonefallow    
   
    
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
        meanT_gt_0 = self.meanT_daily.copy()
        meanT_le_0 = self.meanT_daily.copy()

        meanT_gt_0[meanT_gt_0 <=0] = 0 # removing all negative temperatures for summation
        meanT_le_0[meanT_gt_0 >0] = 0 # removing all positive temperatures for summation 
        ddt = np.sum(meanT_gt_0, axis = 2)
        ddf = - np.sum(meanT_le_0, axis = 2)  
        fi = np.sqrt(ddf)/(np.sqrt(ddf) + np.sqrt(ddt))

        permafrost=fi.copy()
        permafrost=np.where(permafrost>0.625,1,permafrost) # Continuous Permafrost Class
        permafrost=np.where((permafrost>0.57)&(permafrost<0.625),2,permafrost) # Discontinuous Permafrost Class
        permafrost=np.where((permafrost>0.495)&(permafrost<0.57),3,permafrost) # Sporadic Permafrost Class
        permafrost=np.where(permafrost<0.495,4,permafrost) # No Permafrost Class            

        if self.set_mask:
            return [np.ma.masked_where(self.im_mask == 0, fi), np.ma.masked_where(self.im_mask == 0, permafrost)]
        else:
            return [fi, permafrost]

    
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
    
        aez_tclimate = np.zeros((self.im_height, self.im_width), dtype = int)
        
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                
                # keep the tropics lowland
                if tclimate[i_row, i_col] == 1:
                    aez_tclimate[i_row, i_col] = 1
                
                # keep the tropics highland
                if tclimate[i_row, i_col] == 2:
                    aez_tclimate[i_row, i_col] = 2
                
                # grouping all the subtropics climates into a single class 3
                if tclimate[i_row, i_col] == 3:
                    aez_tclimate[i_row, i_col] = 3
                
                if tclimate[i_row, i_col] == 4:
                    aez_tclimate[i_row, i_col] = 3
                
                if tclimate[i_row, i_col] == 5:
                    aez_tclimate[i_row, i_col] = 3
                
                # grouping all the temperate classes into a single class 4
                if tclimate[i_row, i_col] == 6:
                    aez_tclimate[i_row, i_col] = 4
                
                if tclimate[i_row, i_col] == 7:
                    aez_tclimate[i_row, i_col] = 4
                    
                if tclimate[i_row, i_col] == 8:
                    aez_tclimate[i_row, i_col] = 4
                    
                # grouping all the boreal classes into a single class 5
                if tclimate[i_row, i_col] == 9:
                    aez_tclimate[i_row, i_col] = 5
                
                if tclimate[i_row, i_col] == 10:
                    aez_tclimate[i_row, i_col] = 5
                
                if tclimate[i_row, i_col] == 11:
                    aez_tclimate[i_row, i_col] = 5
                
                # changing the arctic class into class 6
                if tclimate[i_row, i_col] == 12:
                    aez_tclimate[i_row, i_col] = 6
        
        # 2nd Step: Classification of Thermal Zones
        aez_tzone = np.zeros((self.im_height, self.im_width), dtype = int)
        
        obj_utilities = UtilitiesCalc.UtilitiesCalc()
        
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                
                meanT_monthly =  obj_utilities.averageDailyToMonthly(self.meanT_daily[i_row, i_col, :])
                # one conditional parameter for temperature accumulation 
                temp_acc_10deg = self.meanT_daily[i_row, i_col, :]
                temp_acc_10deg[temp_acc_10deg < 10] = 0
                
                # Warm Tzone (TZ1)
                if np.logical_or(aez_tclimate[i_row, i_col] == 1, aez_tclimate[i_row, i_col]==3) == True and np.sum(meanT_monthly >= 10) == 12 and np.mean(self.meanT_daily[i_row, i_col, :])>= 20:
                    aez_tzone[i_row, i_col] = 1
                
                # Moderately cool Tzone (TZ2)
                elif np.logical_or(aez_tclimate[i_row, i_col] == 2, aez_tclimate[i_row, i_col]==3) == True and np.sum(meanT_monthly >= 5) == 12 and np.sum(meanT_monthly >=10) >= 8:
                    aez_tzone[i_row, i_col] = 2
                
                # Moderate Tzone (TZ3)
                elif aez_tclimate[i_row, i_col] == 4 and np.sum(meanT_monthly >= 10) >=5 and np.sum(self.meanT_daily[i_row, i_col, :]>20) >= 75 and np.sum(temp_acc_10deg) > 3000:
                    aez_tzone[i_row, i_col] = 3
                
                # Cool Tzone (TZ4)
                elif np.logical_or.reduce((aez_tclimate[i_row, i_col] == 1, aez_tclimate[i_row, i_col] == 2, aez_tclimate[i_row, i_col] == 3, aez_tclimate[i_row, i_col] == 4)) == True and np.sum(meanT_monthly >= 10) >= 4 and np.mean(self.meanT_daily[i_row, i_col, :])>= 0:
                    aez_tzone[i_row, i_col] = 4
                
                # Cold Tzone (TZ5)
                elif np.logical_or.reduce((aez_tclimate[i_row, i_col] == 1, aez_tclimate[i_row, i_col] == 2, aez_tclimate[i_row, i_col] == 3, aez_tclimate[i_row, i_col] == 4, aez_tclimate[i_row, i_col] == 5)) == True and np.sum(meanT_monthly >= 10) <= 3 and np.mean(self.meanT_daily[i_row, i_col, :])>= 0:
                    aez_tzone[i_row, i_col] = 5
                
                # Very cold Tzone (TZ6)
                # else if np.sum(meanT_monthly_sealevel < 10) == 12 or np.mean(self.meanT_daily_sealevel[i_row, i_col, :]) < 0:
                elif np.sum(meanT_monthly < 10) <= 12 and np.mean(self.meanT_daily[i_row, i_col, :])< 0:
                    aez_tzone[i_row, i_col] = 6
        
        
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
        # aez_temp_regime = np.zeros((self.im_height, self.im_width), dtype = int)
        
        # Tropics, lowland (TRC1)
        # aez_temp_regime[np.all([aez_tclimate == 1, aez_tzone==1], axis = 0)] = 1
        
        # Tropics, highland (TRC2)
        # aez_temp_regime[np.all([aez_tclimate == 2, np.logical_or(aez_tzone ==2, aez_tzone == 4)], axis = 0)] = 2
        
        # Subtropics, warm (TRC3)
        # aez_temp_regime[np.all([aez_tclimate == 3, aez_tzone ==1], axis = 0)] = 3
        
        # Sub-tropics, moderately cool (TRC4)
        # aez_temp_regime[np.all([aez_tclimate == 3, aez_tzone ==2], axis = 0)] = 4
        
        # Sub-tropics, cool (TRC5)
        # aez_temp_regime[np.all([aez_tclimate == 3, aez_tzone ==4], axis = 0)] = 5
        
        # Temperate, moderate (TRC6)
        # aez_temp_regime[np.all([aez_tclimate == 4, aez_tzone ==3], axis = 0)] = 6
        
        # Temperate, cool (TRC7)
        # aez_temp_regime[np.all([aez_tclimate == 4, aez_tzone == 4], axis = 0)] = 7
        
        # Boreal, cold (no occurrence of continuous or discontinuous permafrost classes) I (TCR8)
        # aez_temp_regime[np.all([np.logical_or.reduce(np.array((aez_tclimate == 2, aez_tclimate == 3, aez_tclimate == 4, aez_tclimate == 5))), aez_tzone ==5], axis = 0)] = 8
        
        # Boreal, cold (occurrence of continuous or discontinuous permafrost classes) II (TCR9) # need to add permafrost evaluation
        # aez_temp_regime[np.all([np.logical_or.reduce(np.array((aez_tclimate == 2, aez_tclimate == 3, aez_tclimate == 4, aez_tclimate == 5))), aez_tzone ==5], axis = 0)] = 9
        
        # Arctic, very cold
        # aez_temp_regime[np.all([np.logical_or.reduce(np.array((aez_tclimate == 2, aez_tclimate == 3, aez_tclimate == 4, aez_tclimate == 5))), aez_tzone ==6], axis = 0)] = 10
        
        # another way of classification of thermal regime (going pixel by pixel)
        
        aez_temp_regime = np.zeros((self.im_height, self.im_width))

        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                if aez_tclimate[i_row, i_col] == 1 and aez_tzone[i_row, i_col]==1:
                    aez_temp_regime[i_row, i_col]=1 # Tropics, lowland

                elif aez_tclimate[i_row, i_col] == 2 and np.logical_or(aez_tzone[i_row, i_col] == 2, aez_tzone[i_row, i_col] == 4) == True:
                    aez_temp_regime[i_row, i_col] = 2  # Tropics, highland
                    
                elif aez_tclimate[i_row, i_col] == 3 and aez_tzone[i_row, i_col] == 1:
                    aez_temp_regime[i_row, i_col] = 3  # Subtropics, warm
                    
                elif aez_tclimate[i_row, i_col] == 3 and aez_tzone[i_row, i_col] == 2:
                    aez_temp_regime[i_row, i_col] = 4  # Subtropics,moderate cool
                    
                elif aez_tclimate[i_row, i_col] == 3 and aez_tzone[i_row, i_col] == 4:
                    aez_temp_regime[i_row, i_col] = 5  # Subtropics,cool
                    
                elif aez_tclimate[i_row, i_col] == 4 and aez_tzone[i_row, i_col] == 3:
                    aez_temp_regime[i_row, i_col] = 6  # Temperate, moderate
                    
                elif aez_tclimate[i_row, i_col] == 4 and aez_tzone[i_row, i_col] == 4:
                    aez_temp_regime[i_row, i_col] = 7  # Temperate, cool
                    
                elif np.logical_or.reduce((aez_tclimate[i_row, i_col] == 2, aez_tclimate[i_row, i_col] == 3, aez_tclimate[i_row, i_col] == 4, aez_tclimate[i_row, i_col] == 5))== True and aez_tzone[i_row, i_col] == 5 and np.logical_or(permafrost[i_row, i_col] == 1, permafrost[i_row, i_col] == 2) == False:
                    aez_temp_regime[i_row, i_col] = 8  # Boreal/Cold, no permafrost

                elif np.logical_or.reduce((aez_tclimate[i_row, i_col] == 2, aez_tclimate[i_row, i_col] == 3, aez_tclimate[i_row, i_col] == 4, aez_tclimate[i_row, i_col] == 5))== True and aez_tzone[i_row, i_col] == 5 and np.logical_or(permafrost[i_row, i_col] == 1, permafrost[i_row, i_col] == 2) == True:
                    aez_temp_regime[i_row, i_col] = 9  # Boreal/Cold, with permafrost
                    
                elif np.logical_or.reduce((aez_tclimate[i_row, i_col] == 2, aez_tclimate[i_row, i_col] == 3, aez_tclimate[i_row, i_col] == 4, aez_tclimate[i_row, i_col] == 5, aez_tclimate[i_row, i_col] == 6))== True and aez_tzone[i_row, i_col] == 6:
                    aez_temp_regime[i_row, i_col] = 10  # Arctic/Very Cold
                    
        # 4th Step: Moisture Regime classes
        # Moisture Regime Class Definition
        # 1 = M1 (desert/arid areas, 0 <= LGP* < 60)
        # 2 = M2 (semi-arid/dry areas, 60 <= LGP* < 180)
        # 3 = M3 (sub-humid/moist areas, 180 <= LGP* < 270)
        # 4 = M4 (humid/wet areas, LGP* >= 270)
        
        aez_moisture_regime = np.zeros((self.im_height, self.im_width), dtype = int)
        
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                # check if LGP t>5 is greater or less than 330 days. If greater, LGP will be used; otherwise, LGP_equv will be used.
                if lgpt_5[i_row, i_col] > 330:
                    
                    # Class 1 (M1)
                    if lgp[i_row, i_col] >= 0 and lgp[i_row, i_col] < 60:
                        aez_moisture_regime[i_row, i_col] = 1
                    
                    # Class 2 (M2)
                    elif lgp[i_row, i_col] >= 60 and lgp[i_row, i_col] < 180:
                        aez_moisture_regime[i_row, i_col] = 2
                        
                    #Class 3 (M3)   
                    elif lgp[i_row, i_col] >= 180 and lgp[i_row, i_col] < 270:
                        aez_moisture_regime[i_row, i_col] = 3
                        
                    # Class 4 (M4)
                    elif lgp[i_row, i_col] >= 270:
                        aez_moisture_regime[i_row, i_col] = 4
                
                elif lgpt_5[i_row, i_col] <= 330:
                    
                    # Class 1 (M1)
                    if lgp_equv[i_row, i_col] >= 0 and lgp_equv[i_row, i_col] < 60:
                        aez_moisture_regime[i_row, i_col] = 1
                    
                    # Class 2 (M2)
                    elif lgp_equv[i_row, i_col] >= 60 and lgp_equv[i_row, i_col] < 180:
                        aez_moisture_regime[i_row, i_col] = 2
                        
                    #Class 3 (M3)   
                    elif lgp_equv[i_row, i_col] >= 180 and lgp_equv[i_row, i_col] < 270:
                        aez_moisture_regime[i_row, i_col] = 3
                        
                    # Class 4 (M4)
                    elif lgp_equv[i_row, i_col] >= 270:
                        aez_moisture_regime[i_row, i_col] = 4
                
        # Now, we will classify the agro-ecological zonation
        # By GAEZ v4 Documentation, there are prioritized sequential assignment of AEZ classes in order to ensure the consistency of classification 
        aez = np.zeros((self.im_height, self.im_width), dtype = int)
        
        
        for i_row in range(self.im_height):
            for i_col in range(self.im_width):
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
                
                # if it's urban built-up lulc
                if soil_terrain_lulc[i_row, i_col]== 8:
                    aez[i_row, i_col] = 56
                
                # if it's water
                elif soil_terrain_lulc[i_row, i_col]== 7:
                    aez[i_row, i_col] = 57
                
                # if it's dominantly very steep terrain
                elif soil_terrain_lulc[i_row, i_col]== 1:
                    aez[i_row, i_col] = 49
                
                # if it's irrigated soils
                elif soil_terrain_lulc[i_row, i_col]== 6:
                    aez[i_row, i_col] = 51
                
                # if it's hydromorphic soils
                elif soil_terrain_lulc[i_row, i_col]== 2:
                    aez[i_row, i_col] = 52
                    
                elif aez_moisture_regime[i_row, i_col]==1:
                    aez[i_row, i_col] = 53
                    
                elif aez_temp_regime[i_row, i_col]==9 and aez_moisture_regime[i_row, i_col] in [1,2,3,4] == True:
                    aez[i_row, i_col] = 54
                    
                elif soil_terrain_lulc[i_row, i_col]==5:
                    aez[i_row, i_col] = 50
                
                elif aez_temp_regime[i_row, i_col]==10 and aez_moisture_regime[i_row, i_col] in [1,2,3,4] == True:
                    aez[i_row, i_col] = 55
                    
                #######
                elif aez_temp_regime[i_row, i_col]==1 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 1
                
                elif aez_temp_regime[i_row, i_col]==1 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 2
                
                elif aez_temp_regime[i_row, i_col]==1 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 3
                
                elif aez_temp_regime[i_row, i_col]==1 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 4
                
                elif aez_temp_regime[i_row, i_col]==1 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 5
                
                elif aez_temp_regime[i_row, i_col]==1 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 6
                ####
                elif aez_temp_regime[i_row, i_col]==2 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 7
                
                elif aez_temp_regime[i_row, i_col]==2 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 8
                
                elif aez_temp_regime[i_row, i_col]==2 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 9
                
                elif aez_temp_regime[i_row, i_col]==2 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 10
                
                elif aez_temp_regime[i_row, i_col]==2 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 11
                
                elif aez_temp_regime[i_row, i_col]==2 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 12
                ###
                elif aez_temp_regime[i_row, i_col]==3 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 13
                
                elif aez_temp_regime[i_row, i_col]==3 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 14
                
                elif aez_temp_regime[i_row, i_col]==3 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 15
                
                elif aez_temp_regime[i_row, i_col]==3 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 16
                
                elif aez_temp_regime[i_row, i_col]==3 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 17
                
                elif aez_temp_regime[i_row, i_col]==3 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 18
                #####
                elif aez_temp_regime[i_row, i_col]==4 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 19
                
                elif aez_temp_regime[i_row, i_col]==4 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 20
                
                elif aez_temp_regime[i_row, i_col]==4 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 21
                
                elif aez_temp_regime[i_row, i_col]==4 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 22
                
                elif aez_temp_regime[i_row, i_col]==4 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 23
                
                elif aez_temp_regime[i_row, i_col]==4 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 24
                #####
                elif aez_temp_regime[i_row, i_col]==5 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 25
                
                elif aez_temp_regime[i_row, i_col]==5 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 26
                
                elif aez_temp_regime[i_row, i_col]==5 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 27
                
                elif aez_temp_regime[i_row, i_col]==5 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 28
                
                elif aez_temp_regime[i_row, i_col]==5 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 29
                
                elif aez_temp_regime[i_row, i_col]==5 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 30
                ######
                
                elif aez_temp_regime[i_row, i_col]==6 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 31
                
                elif aez_temp_regime[i_row, i_col]==6 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 32
                
                elif aez_temp_regime[i_row, i_col]==6 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 33
                
                elif aez_temp_regime[i_row, i_col]==6 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 34
                
                elif aez_temp_regime[i_row, i_col]==6 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 35
                
                elif aez_temp_regime[i_row, i_col]==6 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 36
                
                ###
                elif aez_temp_regime[i_row, i_col]==7 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 37
                
                elif aez_temp_regime[i_row, i_col]==7 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 38
                
                elif aez_temp_regime[i_row, i_col]==7 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 39
                
                elif aez_temp_regime[i_row, i_col]==7 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 40
                
                elif aez_temp_regime[i_row, i_col]==7 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 41
                
                elif aez_temp_regime[i_row, i_col]==7 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 42
                #####
                
                elif aez_temp_regime[i_row, i_col]==8 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 43
                
                elif aez_temp_regime[i_row, i_col]==8 and aez_moisture_regime[i_row, i_col]==2 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 44
                
                elif aez_temp_regime[i_row, i_col]==8 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 45
                
                elif aez_temp_regime[i_row, i_col]==8 and aez_moisture_regime[i_row, i_col]==3 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 46
                
                elif aez_temp_regime[i_row, i_col]==8 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==3:
                    aez[i_row, i_col] = 47
                
                elif aez_temp_regime[i_row, i_col]==8 and aez_moisture_regime[i_row, i_col]==4 and soil_terrain_lulc[i_row, i_col]==4:
                    aez[i_row, i_col] = 48
                

        if self.set_mask:
            return np.ma.masked_where(self.im_mask==0, aez)
        else:        
            return aez
    
    
#----------------- End of file -------------------------#