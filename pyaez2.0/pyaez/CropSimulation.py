"""
PyAEZ version 2.0.0 (November 2022)
This CropSimulation Class simulates all the possible crop cycles to find 
the best crop cycle that produces maximum yield for a particular grid
2020: N. Lakmal Deshapriya
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
try:
    import gdal
except:
    from osgeo import gdal

import UtilitiesCalc
import BioMassCalc
import ETOCalc
import CropWatCalc
import ThermalScreening

class CropSimulation(object):

    def __init__(self):
        """Initiate a Class instance
        """        
        self.set_mask = False
        self.set_tclimate_screening = False
        self.set_lgpt_screening = False
        self.set_Tsum_screening = False
        self.set_Tprofile_screening = False

    def setMonthlyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):
        """Read and load the MONTHLY climate data into the Class
        Args:
            min_temp (3D NumPy): Monthly minimum temperature [Celcius]
            max_temp (3D NumPy): Monthly maximum temperature [Celcius]
            precipitation (3D NumPy): Monthly total precipitation [mm/day]
            short_rad (3D NumPy): Monthly solar radiation [W/m2]
            wind_speed (3D NumPy): Monthly windspeed at 2m altitude [m/s]
            rel_humidity (3D NumPy): Monthly relative humidity [percentage decimal, 0-1]
        """
        rel_humidity[rel_humidity > 0.99] = 0.99
        rel_humidity[rel_humidity < 0.05] = 0.05
        short_rad[short_rad < 0] = 0
        wind_speed[wind_speed < 0] = 0
        self.minT_monthly = min_temp
        self.maxT_monthly = max_temp
        self.totalPrec_monthly = precipitation
        self.shortRad_monthly = short_rad
        self.wind2m_monthly = wind_speed
        self.rel_humidity_monthly = rel_humidity
        self.set_monthly = True



    def setDailyClimateData(self, min_temp, max_temp, precipitation, short_rad, wind_speed, rel_humidity):
        """Load DAILY climate data into the Class

        Args:
            min_temp (3D NumPy): Daily minimum temperature [Celcius]
            max_temp (3D NumPy): Daily maximum temperature [Celcius]
            precipitation (3D NumPy): Daily total precipitation [mm/day]
            short_rad (3D NumPy): Daily solar radiation [W/m2]
            wind_speed (3D NumPy): Daily windspeed at 2m altitude [m/s]
            rel_humidity (3D NumPy): Daily relative humidity [percentage decimal, 0-1]
        """
        rel_humidity[rel_humidity > 0.99] = 0.99
        rel_humidity[rel_humidity < 0.05] = 0.05
        short_rad[short_rad < 0] = 0
        wind_speed[wind_speed < 0] = 0

        self.minT_daily = min_temp
        self.maxT_daily = max_temp
        self.totalPrec_daily = precipitation
        self.shortRad_daily = short_rad
        self.wind2m_daily = wind_speed
        self.rel_humidity_daily = rel_humidity
        self.set_monthly = False

    def setLocationTerrainData(self, lat_min, lat_max, elevation):
        """Load geographical extents and elevation data in to the Class, 
           and create a latitude map

        Args:
            lat_min (float): the minimum latitude of the AOI in decimal degrees
            lat_max (float): the maximum latitude of the AOI in decimal degrees
            elevation (2D NumPy): elevation map in metres
        """
        self.elevation = elevation
        self.im_height = elevation.shape[0]
        self.im_width = elevation.shape[1]
        self.latitude = UtilitiesCalc.UtilitiesCalc().generateLatitudeMap(lat_min, lat_max, self.im_height, self.im_width)

    def setCropParameters(self, LAI, HI, legume, adaptability, cycle_len, D1, D2):
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
        self.LAi = LAI # leaf area index
        self.HI = HI # harvest index
        self.legume = legume # binary value
        self.adaptability = adaptability  #one of [1,2,3,4] classes
        self.cycle_len = cycle_len  #length of growing period

        self.D1 = D1  # rooting depth 1 (m)
        self.D2 = D2  # rooting depth 2 (m)

    def setCropCycleParameters(self, stage_per, kc, kc_all, yloss_f, yloss_f_all):
        """This function allow user to set up the parameters related to the crop cyles.

        Args:
            stage_per (4-item list): Percentage of each 4 stages of a crop cycle,
            kc (3-item list): crop water requirements for initial, reproductive, the end of the maturation stage
            kc_all (float): Crop water requirements for entire growth cycle.
            yloss_f (4-item list): Yield loss factor of each 4 stages of a crop cycle
            yloss_f_all (4-item list): Yield loss factor for the entire growth cycle

        """        
        self.d_per = stage_per # Percentage for D1, D2, D3, D4 stages
        self.kc = kc # 3 crop water requirements for initial, reproductive, the end of the maturation stages
        self.kc_all = kc_all # crop water requirements for entire growth cycle
        self.yloss_f = yloss_f  # yield loss for D1, D2, D3, D4
        self.yloss_f_all = yloss_f_all  # yield loss for entire growth cycle

    def setSoilWaterParameters(self, Sa, pc):
        """This function allow user to set up the parameters related to the soil water storage.

        Args:
            Sa (float or 2D numpy): Available  soil moisture holding capacity
            pc (float): Soil water depletion fraction below which ETa<ETo
        """        
        self.Sa = Sa  # available soil moisture holding capacity (mm/m) , assumption
        self.pc = pc  # soil water depletion fraction below which ETa < ETo (from literature)

    '''All of bellow settings are optional'''

    # set mask of study area, this is optional
    def setStudyAreaMask(self, admin_mask, no_data_value):
        """Set clipping mask of the area of interest (optional)

        Args:
            admin_mask (2D NumPy/Binary): mask to extract only region of interest
            no_data_value (int): pixels with this value will be omitted during PyAEZ calculations
        """
        self.im_mask = admin_mask
        self.nodata_val = no_data_value
        self.set_mask = True

    def adjustForPerennialCrop(self,  aLAI, bLAI, aHI, bHI):
        """If a perennial crop is introduced, PyAEZ will perform adjustment 
        on the Leaf Area Index (LAI) and the Harvest Index (HI) based 
        on the effective growing period.

        Args:
            aLAI (int): alpha coefficient for LAI
            bLAI (int): beta coefficient for LAI
            aHI (int): alpha coefficient for HI
            bHI (int): beta coefficient for HI
        """        
        self.LAi = self.LAi * ((self.cycle_len-aLAI)/bLAI) # leaf area index adjustment for perennial crops
        self.HI = self.HI * ((self.cycle_len-aHI)/bHI) # harvest index adjustment for perennial crops

    def setThermalClimateScreening(self, t_climate, no_t_climate):
        """Load thermal climate into the Class 

        Args:
            t_climate (2D NumPy): Thermal Climate classes
            no_t_climate (list): Pixels values of 'not suitable' thermal climate classes
        """        
        self.t_climate = t_climate
        self.no_t_climate = no_t_climate # list of unsuitable thermal climate

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

    def setTSumScreening(self, no_Tsum, optm_Tsum):
        """Set screening parameters for thermal summation(TS)

        Args:
            no_Tsum (3-item list): 3 'not suitable' TS conditions
            optm_Tsum (3-item list): 3 'optimum' TS conditions
        """
        self.no_Tsum = no_Tsum
        self.optm_Tsum = optm_Tsum

        self.set_Tsum_screening = True

    def setTProfileScreening(self, no_Tprofile, optm_Tprofile):
        """Set screening parameters for temperature profile (Tprofile)

        Args:
            no_Tprofile (18-item list): 18 'not suitable' Tprofile conditions
            optm_Tprofile (18-item list): 18 'optimum' Tprofile conditions
        """        
        self.no_Tprofile = no_Tprofile
        self.optm_Tprofile = optm_Tprofile

        self.set_Tprofile_screening = True

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

        # this stores final result
        self.final_yield_rainfed = np.zeros((self.im_height, self.im_width))
        self.final_yield_irrig = np.zeros((self.im_height, self.im_width))
        self.crop_calender = np.zeros((self.im_height, self.im_width))

        for i_row in range(self.im_height):

            if self.set_mask:
                print('Done: ' + str(int((count_pixel_completed /
                      np.sum(self.im_mask != self.nodata_val))*100)) + ' %')
            else:
                print('Done: ' + str(int((count_pixel_completed /
                      (self.im_height*self.im_width))*100)) + ' %')

            for i_col in range(self.im_width):

                # check current location (pixel) is outside of study area or not. if it's outside of study area goes to next location (pixel)
                if self.set_mask:
                    if self.im_mask[i_row, i_col] == self.nodata_val:
                        continue
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

                # calculate ETO for full year for particular location (pixel)
                obj_eto = ETOCalc.ETOCalc(
                    1, minT_daily_point.shape[0], self.latitude[i_row, i_col], self.elevation[i_row, i_col])
                shortRad_dailyy_point_MJm2day = (
                    shortRad_daily_point*3600*24)/1000000  # convert w/m2 to MJ/m2/day
                obj_eto.setClimateData(minT_daily_point, maxT_daily_point, wind2m_daily_point,
                                       shortRad_dailyy_point_MJm2day, rel_humidity_daily_point)
                pet_daily_point = obj_eto.calculateETO()

                # list that stores yield estimations of all cycles per particular location (pixel)
                yield_of_all_crop_cycles_rainfed = []
                yield_of_all_crop_cycles_irrig = []

                for i_cycle in range(start_doy, end_doy+1, step_doy):

                    # just repeat data for year 2 times, to simulate for a entire year. just for computational convenient
                    minT_daily_2year = np.tile(minT_daily_point, 2)
                    maxT_daily_2year = np.tile(maxT_daily_point, 2)
                    shortRad_daily_2year = np.tile(shortRad_daily_point, 2)
                    wind2m_daily_2year = np.tile(wind2m_daily_point, 2)
                    totalPrec_daily_2year = np.tile(totalPrec_daily_point, 2)
                    pet_daily_2year = np.tile(pet_daily_point, 2)

                    # extract climate data within the season to pass in to calculation classes
                    minT_daily_season = minT_daily_2year[i_cycle: i_cycle+self.cycle_len]
                    maxT_daily_season = maxT_daily_2year[i_cycle: i_cycle+self.cycle_len]
                    shortRad_daily_season = shortRad_daily_2year[i_cycle: i_cycle+self.cycle_len]
                    wind2m_daily_season = wind2m_daily_2year[i_cycle: i_cycle+self.cycle_len]
                    totalPrec_daily_season = totalPrec_daily_2year[i_cycle: i_cycle+self.cycle_len]
                    pet_daily_season = pet_daily_2year[i_cycle: i_cycle+self.cycle_len]

                    # conduct tests to check simulation should be carried out or not based on growing period threshold. if not, goes to next location (pixel)
                    obj_screening = ThermalScreening.ThermalScreening()
                    obj_screening.setClimateData(
                        minT_daily_season, maxT_daily_season)

                    if self.set_tclimate_screening:
                        obj_screening.setThermalClimateScreening(
                            self.t_climate[i_row, i_col], self.no_t_climate)
                    if self.set_lgpt_screening:
                        obj_screening.setLGPTScreening(
                            self.no_lgpt, self.optm_lgpt)
                    if self.set_Tsum_screening:
                        obj_screening.setTSumScreening(
                            self.no_Tsum, self.optm_Tsum)
                    if self.set_Tprofile_screening:
                        obj_screening.setTProfileScreening(
                            self.no_Tprofile, self.optm_Tprofile)

                    thermal_screening_f = 1
                    if not obj_screening.getSuitability():
                        continue
                    else:
                        thermal_screening_f = obj_screening.getReductionFactor()

                    # calculate biomass
                    obj_maxyield = BioMassCalc.BioMassCalc(
                        i_cycle, i_cycle+self.cycle_len-1, self.latitude[i_row, i_col])
                    obj_maxyield.setClimateData(
                        minT_daily_season, maxT_daily_season, shortRad_daily_season)
                    obj_maxyield.setCropParameters(
                        self.LAi, self.HI, self.legume, self.adaptability)
                    obj_maxyield.calculateBioMass()
                    est_yield = obj_maxyield.calculateYield()

                    # reduce thermal screening factor
                    est_yield = est_yield * thermal_screening_f

                    # apply cropwat
                    obj_cropwat = CropWatCalc.CropWatCalc(
                        i_cycle, i_cycle+self.cycle_len-1)
                    obj_cropwat.setClimateData(
                        pet_daily_season, totalPrec_daily_season)
                    # check Sa is a raster or single value and extract Sa value accordingly
                    if len(np.array(self.Sa).shape) == 2:
                        Sa_temp = self.Sa[i_row, i_col]
                    else:
                        Sa_temp = self.Sa
                    obj_cropwat.setCropParameters(
                        self.d_per, self.kc, self.kc_all, self.yloss_f, self.yloss_f_all, est_yield, self.D1, self.D2, Sa_temp, self.pc)
                    est_yield_moisture_limited = obj_cropwat.calculateMoistureLimitedYield()

                    # append current cycle yield to a list
                    yield_of_all_crop_cycles_rainfed.append(
                        est_yield_moisture_limited)
                    yield_of_all_crop_cycles_irrig.append(est_yield)

                    # get maximum yield from all simulation for a particular location (pixel) and assign to final map
                    if len(yield_of_all_crop_cycles_irrig) > 0:
                        self.final_yield_rainfed[i_row, i_col] = np.max(
                            yield_of_all_crop_cycles_rainfed)
                        self.final_yield_irrig[i_row, i_col] = np.max(
                            yield_of_all_crop_cycles_irrig)
                        self.crop_calender[i_row, i_col] = np.where(yield_of_all_crop_cycles_rainfed == np.max(
                            yield_of_all_crop_cycles_rainfed))[0][0] * step_doy

        print('Simulations Completed !')

    def getEstimatedYieldRainfed(self):
        """Estimation of Maximum Yield for Rainfed scenario

        Returns:
            2D NumPy: the maximum attainable yield under the provided climate conditions, 
                      under rain-fed conditions [kg/ha]
        """   
        if self.set_mask:
            return np.ma.masked_where(self.im_mask==0,self.final_yield_rainfed)   
        else:  
            return self.final_yield_rainfed

    def getEstimatedYieldIrrigated(self):
        """Estimation of Maximum Yield for Irrigated scenario

        Returns:
            2D NumPy: the maximum attainable yield under the provided climate conditions, 
                      under irrigated conditions [kg/ha]
        """
        if self.set_mask:
            return np.ma.masked_where(self.im_mask==0,self.final_yield_irrig)   
        else:  
            return self.final_yield_irrig

    def getOptimumCycleStartDate(self):
        """Optimum starting date for crop cycle

        Returns:
            2D NumPy: Each pixel value corresponds to the Julian day of the optimum crop cycle starting date.
        """
        if self.set_mask:
            return np.ma.masked_where(self.im_mask==0,self.crop_calender)   
        else:  
            return self.crop_calender
        

#----------------- End of file -------------------------#