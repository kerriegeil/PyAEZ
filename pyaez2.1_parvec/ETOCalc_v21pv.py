"""
PyAEZ version 2.1.0 (June 2023)
ETOCalc.py calculates the reference evapotranspiration from 
the climatic data provided by the PyAEZ user.
2020: N. Lakmal Deshapriya, Thaileng Thol
2022/2023: Kittiphon Boonma  (Numba)

"""
import numpy as np
import numba as nb

class ETOCalc(object):
 
    def __init__(self, cycle_begin, cycle_end, latitude, altitude):
        """Initiate a ETOCalc Class instance

        Args:
            cycle_begin (int): Julian day for the beginning of crop cycle
            cycle_end (int): Julian day for the ending of crop cycle
            latitude (float): a latitude value
            altitude (float): an altitude value
        """        
        self.cycle_begin = cycle_begin
        self.cycle_end = cycle_end
        self.latitude = latitude
        self.alt = altitude

    def setClimateData(self, min_temp, max_temp, wind_speed, short_rad, rel_humidity):
        """Load the climatic (point) data into the Class

        Args:
            min_temp (float): Minimum temperature [Celcius]
            max_temp (float): Maximum temperature [Celcius]
            wind_speed (float): Windspeed at 2m altitude [m/s]
            short_rad (float): Radiation [MJ/m2.day]
            rel_humidity (float): Relative humidity [decimal percentage]
        """        

        self.minT_daily = min_temp # Celcius
        self.maxT_daily = max_temp # Celcius
        self.windspeed_daily = wind_speed # m/s at 2m
        self.shortRad_daily = short_rad # MJ/m2.day
        self.rel_humidity = rel_humidity # Fraction

    # @staticmethod
    # @nb.jit(nopython=True)
    # def calculateETONumba(cycle_begin, cycle_end, latitude, alt,  minT_daily, maxT_daily, windspeed_daily, shortRad_daily, rel_humidity):
    def calculateETO(self):  #KLG
        # numba doesn't speed this up in time tests  #KLG
        # removing in favor of vectorization which will allow chunking with dask for speed  #KLG

        """Calculate the reference evapotranspiration with Penmann-Monteith Equation

        Returns:
            ## float: ETo of a single pixel (function is called pixel-wise)
            float: ETo of each pixel  #KLG
        """        
        # constants
        # tavg = 0.5*(maxT_daily+minT_daily)  # Averaged temperature
        tavg = 0.5*(self.minT_daily + self.maxT_daily)  # Averaged temperature  #KLG
        lam = 2.501 - 0.002361 * tavg  # Latent heat of vaporization
        dayoyr = np.arange(self.cycle_begin, self.cycle_end+1)  # Julien Days #KLG
        ndays=len(dayoyr)  #KLG
        alt=np.tile(self.alt[:,:,np.newaxis],(1,1,ndays))  # 3D altitude #KLG

        # Wind speed
        # u2m = windspeed_daily.copy()
        u2m = self.windspeed_daily.copy()  #KLG
        # limit to no less than 0.5 m/s; FAO 56, p.63
        # u2m[windspeed_daily < 0.5] = 0.5
        u2m[self.windspeed_daily < 0.5] = 0.5

        # Mean Saturation Vapor Pressure derived from air temperature
        # es_tmin = 0.6108 * np.exp((17.27 * minT_daily) / (minT_daily + 237.3))
        # es_tmax = 0.6108 * np.exp((17.27 * maxT_daily) / (maxT_daily + 237.3))
        es_tmin = 0.6108 * np.exp((17.27 * self.minT_daily) / (self.minT_daily + 237.3))  #KLG
        es_tmax = 0.6108 * np.exp((17.27 * self.maxT_daily) / (self.maxT_daily + 237.3))  #KLG
        es = 0.5*(es_tmin + es_tmax)
        # ea = rel_humidity * es  # Actual Vapor Pressure derived from relative humidity
        ea = self.rel_humidity * es  # Actual Vapor Pressure derived from relative humidity  #KLG

        # slope vapour pressure curve
        # dlmx = 4098. * es_tmax / (maxT_daily + 237.3)**2
        # dlmn = 4098. * es_tmin / (minT_daily + 237.3)**2
        dlmx = 4098. * es_tmax / (self.maxT_daily + 237.3)**2  #KLG
        dlmn = 4098. * es_tmin / (self.minT_daily + 237.3)**2  #KLG
        del es_tmin, es_tmax  #KLG
        dl = 0.5* (dlmx + dlmn)
        del dlmx,dlmn  #KLG

        # Atmospheric pressure
        ap = 101.3*np.power(((293-(0.0065*alt))/293), 5.256)

        # Psychrometric constant
        gam = 0.0016286 * ap/lam
        del ap

        hw = 200.
        ht = 190.
        hc = 12.

        # aerodynamic resistance
        rhoa = 208/u2m

        # crop canopy resistance
        Rl = 100  # daily stomata resistance of a single leaf (s/m)
        # Standard is xLAO = 24
        RLAI = 24 * 0.12
        rhoc = Rl/(0.5*RLAI)  # crop canopy resistance

        gamst = gam * (1. + rhoc/rhoa)

        # net radiation Rn = Rns - Rnl
        # Julien Days
        # dayoyr = np.arange(cycle_begin, cycle_end+1)

        # latr = latitude * np.pi/180.
        latr = self.latitude * np.pi/180.
        latr = np.tile(latr[:,:,np.newaxis],(1,1,ndays))

        # (a) calculate extraterrestrial radiation
        # solar declination (rad)
        sdcl = 0.4093 * np.sin(0.017214206 * dayoyr - 1.405)
        sdcl = np.tile(sdcl[np.newaxis, np.newaxis,:], (tavg.shape[0],tavg.shape[1],1))
        # relative distance earth to sun
        sdst = 1.0 + 0.033 * np.cos(0.017214206 * dayoyr)
        sdst=np.tile(sdst[np.newaxis, np.newaxis,:], (tavg.shape[0],tavg.shape[1],1))
        del dayoyr
        xx = np.sin(sdcl) * np.sin(latr)
        yy = np.cos(sdcl) * np.cos(latr)
        zz = xx/yy
        # omg = np.arccos(-np.tan(latr)*np.tan(sdcl))  # Sunset hour angle (rad)

        omg = np.tan(zz / (1. - zz*zz)**0.5) + 1.5708
        dayhr = 24. * (omg/np.pi)

        omg[np.where(np.logical_and(np.abs(zz) >= 0.9999, zz > 0))] = np.pi
        dayhr[np.where(np.logical_and(np.abs(zz) >= 0.9999, zz > 0))] = 23.999

        omg[np.where(np.logical_and(np.abs(zz) >= 0.9999, zz <= 0))] = 0
        dayhr[np.where(np.logical_and(np.abs(zz) >= 0.9999, zz <= 0))] = 0.001

        ra = 37.586 * sdst * (omg*xx + np.sin(omg)*yy)
        del sdcl, latr, sdst, omg, dayhr, xx, yy, zz        

        # (b) solar radiation Rs (0.25, 0.50 Angstrom coefficients)
        # rs = (0.25 + (0.50 * (sd/dayhr))) * ra
        # rs = shortRad_daily
        rs = self.shortRad_daily  #KLG
        rs0 = (0.75 + 0.00002 * alt) * ra
        del alt

        # (c) net shortwave radiation Rns = (1 - alpha) * Rs
        # (alpha for grass = 0.23)
        rns = 0.77 * rs

        # (d) net longwave radiation Rnl
        # Stefan-Boltzmann constant [MJ K-4 m-2 day-1]
        sub_cst = 0.000000004903
        # rnl = sub_cst * (0.1 + 0.9 * (sd / dayhr)) * (0.34 - 0.139 * np.sqrt(ea)) * \
        #     0.5 * ((maxT_daily + 273.16) **
        #            4 + (minT_daily + 273.16) ** 4)
        # Stefan-Boltzmann constant [MJ K-4 m-2 day-1]
        # rnl = (((273.16+maxT_daily)**4)+((273.16 + minT_daily)**4)) * \
        #     (0.34 - (0.14*(ea**0.5))) * \
        #     ((1.35*(rs/rs0))-0.35)*sub_cst/2
        rnl = (((273.16+self.maxT_daily)**4)+((273.16 + self.minT_daily)**4)) * \
            (0.34 - (0.14*(ea**0.5))) * \
            ((1.35*(rs/rs0))-0.35)*sub_cst/2
        del rs0,rs

        # (e) net radiation Rn = Rns - Rnl
        rn = rns - rnl
        del rns,rnl
        # rn0 = rn

        # (f) soil heat flux [MJ/m2/day]
        # ta_dublicate_last2 = np.append(tavg, np.array([tavg[-1]]))
        # ta_dublicate_first2 = np.append(np.array([tavg[-1]]), tavg)
        # G = 0.14 * (ta_dublicate_last2 - ta_dublicate_first2)
        # G = G[0:G.size-1]
        ta_diff=np.diff(tavg,n=1,axis=2,prepend=tavg[:,:,-1:])  #KLG
        G = 0.14 * ta_diff  #KLG
        del ta_diff
        # G = 0

        # (g) calculate aerodynamic and radiation terms of ET0
        et0ady = gam/(dl+gamst) * 900./(tavg+273.) * u2m * (es-ea)
        del gam,tavg,u2m,es,ea

        et0rad = dl/(dl+gamst) * (rn-G) / lam
        del dl,gamst,rn,G,lam

        et0 = et0ady + et0rad
        del et0ady,et0rad

        et0[et0 < 0] = 0

        return et0

    # def calculateETO(self):
    #     return ETOCalc.calculateETONumba(self.cycle_begin, self.cycle_end, self.latitude, self.alt,  self.minT_daily, self.maxT_daily, self.windspeed_daily, self.shortRad_daily, self.rel_humidity)
