"""
PyAEZ version 2.1.0 (June 2023)
ETOCalc.py calculates the reference evapotranspiration from 
the climatic data provided by the PyAEZ user.
2020: N. Lakmal Deshapriya, Thaileng Thol
2022/2023: Kittiphon Boonma  (Numba)

"""
import numpy as np
# import numba as nb

class ETOCalc(object):
    # def __init__(self, cycle_begin, cycle_end, latitude, altitude):
    #     """Initiate a ETOCalc Class instance

    #     Args:
    #         cycle_begin (int): Julian day for the beginning of crop cycle
    #         cycle_end (int): Julian day for the ending of crop cycle
    #         latitude (float): a latitude value
    #         altitude (float): an altitude value
    #     """        
    #     self.cycle_begin = cycle_begin
    #     self.cycle_end = cycle_end
    #     self.latitude = latitude
    #     self.alt = altitude

    # def setClimateData(self, min_temp, max_temp, wind_speed, short_rad, rel_humidity):
    #     """Load the climatic (point) data into the Class

    #     Args:
    #         min_temp (float): Minimum temperature [Celcius]
    #         max_temp (float): Maximum temperature [Celcius]
    #         wind_speed (float): Windspeed at 2m altitude [m/s]
    #         short_rad (float): Radiation [MJ/m2.day]
    #         rel_humidity (float): Relative humidity [decimal percentage]
    #     """        

    #     self.minT_daily = min_temp # Celcius
    #     self.maxT_daily = max_temp # Celcius
    #     self.windspeed_daily = wind_speed # m/s at 2m
    #     self.shortRad_daily = short_rad # MJ/m2.day
    #     self.rel_humidity = rel_humidity # Fraction

    # @staticmethod
    # @nb.jit(nopython=True)
    # def calculateETONumba(cycle_begin, cycle_end, latitude, alt,  minT_daily, maxT_daily, windspeed_daily, shortRad_daily, rel_humidity):
    def calculateETO(self,dstart,dend,cdims,lat,alt,tmn,tmx,u2m,srad,rh):  #KLG
        # numba doesn't speed this up in time tests  #KLG
        # removing in favor of vectorization which will allow chunking with dask for speed  #KLG

        """Calculate the reference evapotranspiration with Penmann-Monteith Equation

        Returns:
            ## float: ETo of a single pixel (function is called pixel-wise)
            float: ETo of each pixel  #KLG
        """        
        # constants
        nlats=cdims[0]
        nlons=cdims[1]
    
        tavg = 0.5*(tmn + tmx)  # Averaged temperature  #KLG
        lam = 2.501 - 0.002361 * tavg  # Latent heat of vaporization
        dayoyr = np.arange(dstart, dend+1)  # Julien Days #KLG
        ndays=len(dayoyr)  #KLG

        # Mean Saturation Vapor Pressure derived from air temperature
        es_tmin = 0.6108 * np.exp((17.27 * tmn) / (tmn + 237.3))
        es_tmax = 0.6108 * np.exp((17.27 * tmx) / (tmx + 237.3))
        es = 0.5*(es_tmin + es_tmax)
        ea = rh * es  # Actual Vapor Pressure derived from relative humidity

        # slope vapour pressure curve
        dlmx = 4098. * es_tmax / (tmx + 237.3)**2
        dlmn = 4098. * es_tmin / (tmn + 237.3)**2
        del es_tmin, es_tmax  #KLG
        dl = 0.5* (dlmx + dlmn)
        del dlmx,dlmn  #KLG

        # Atmospheric pressure
        ap = 101.3*np.power(((293-(0.0065*alt))/293), 5.256)
        ap = np.tile(ap[:,:,np.newaxis],(1,1,ndays)) 

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


        # latr = latitude * np.pi/180.
        latr = lat * np.pi/180.
        latr = np.tile(latr[:,:,np.newaxis],(1,1,ndays))

        # (a) calculate extraterrestrial radiation
        # solar declination (rad)
        sdcl = 0.4093 * np.sin(0.017214206 * dayoyr - 1.405)
        sdcl = np.tile(sdcl[np.newaxis, np.newaxis,:], (nlats,nlons,1))

        # relative distance earth to sun
        xx = np.sin(sdcl) * np.sin(latr)
        yy = np.cos(sdcl) * np.cos(latr)
        zz = xx/yy        

        with np.errstate(invalid='ignore'):
            omg = np.tan(zz / (1. - zz*zz)**0.5) + 1.5708
        dayhr = 24. * (omg/np.pi)

        omg=np.where((np.abs(zz)>=0.9999)&(zz>0),np.pi,omg)
        dayhr=np.where((np.abs(zz) >= 0.9999)&(zz > 0), 23.999,dayhr)
        omg=np.where((np.abs(zz) >= 0.9999)&(zz <= 0),0,omg)
        dayhr=np.where((np.abs(zz) >= 0.9999)&(zz <= 0), 0.001,dayhr)
        del latr,sdcl,zz        

        sdst = 1.0 + 0.033 * np.cos(0.017214206 * dayoyr)
        sdst = np.tile(sdst[np.newaxis, np.newaxis,:], (nlats,nlons,1))
        del dayoyr

        ra = 37.586 * sdst * (omg*xx + np.sin(omg)*yy)
        del sdst, omg, dayhr, xx, yy        

        # (b) solar radiation Rs (0.25, 0.50 Angstrom coefficients)
        # rs = (0.25 + (0.50 * (sd/dayhr))) * ra
        # rs = shortRad_daily
        rs = srad
        alt=np.tile(alt[:,:,np.newaxis],(1,1,ndays)) 
        rs0 = (0.75 + 0.00002 * alt) * ra
        del alt

        # (c) net shortwave radiation Rns = (1 - alpha) * Rs
        # (alpha for grass = 0.23)
        rns = 0.77 * rs

        # (d) net longwave radiation Rnl
        # Stefan-Boltzmann constant [MJ K-4 m-2 day-1]
        sub_cst = 0.000000004903
        with np.errstate(invalid='ignore',divide='ignore'):
            rs_div_ds0=rs/rs0
        rnl = (((273.16+tmx)**4)+((273.16 + tmn)**4)) * \
            (0.34 - (0.14*(ea**0.5))) * \
            ((1.35*(rs_div_ds0))-0.35)*sub_cst/2
        del rs0,rs,rs_div_ds0,tmx,tmn

        # (e) net radiation Rn = Rns - Rnl
        rn = rns - rnl
        del rns,rnl

        # (f) soil heat flux [MJ/m2/day]
        ta_diff=np.diff(tavg,n=1,axis=2,prepend=tavg[:,:,-1:])  #KLG
        G = 0.14 * ta_diff  #KLG
        del ta_diff

        # (g) calculate aerodynamic and radiation terms of ET0
        et0ady = gam/(dl+gamst) * 900./(tavg+273.) * u2m * (es-ea)
        del gam,tavg,u2m,es,ea

        et0rad = dl/(dl+gamst) * (rn-G) / lam
        del dl,gamst,rn,G,lam

        et0 = et0ady + et0rad
        del et0ady,et0rad

        et0 = np.where(et0<0,0,et0)

        return et0.astype('float32')

    # def calculateETO(self):
    #     return ETOCalc.calculateETONumba(self.cycle_begin, self.cycle_end, self.latitude, self.alt,  self.minT_daily, self.maxT_daily, self.windspeed_daily, self.shortRad_daily, self.rel_humidity)
