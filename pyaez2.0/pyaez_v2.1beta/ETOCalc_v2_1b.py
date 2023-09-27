"""
ETOCalc.py calculates the reference evapotranspiration from 
the climatic data provided by the PyAEZ user.
2020: N. Lakmal Deshapriya
2022/2023: Kittiphon Boonma - converted to Numba

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

    @staticmethod
    @nb.jit(nopython=True)
    def calculateETONumba(cycle_begin, cycle_end, latitude, alt,  minT_daily, maxT_daily, windspeed_daily, shortRad_daily, rel_humidity):
        """Calculate the reference evapotranspiration with Penmann-Monteith Equation

        Returns:
            float: ETo of a single pixel (function is called pixel-wise)
        """        
        # (1) Monthly Average Temperature
        ta = 0.5*(minT_daily + maxT_daily)
        # Latent heat of vaporization
        lam = 2.501-0.002361*ta
        # (2) Mean daily solar radiation
        # Rs (MJ/m2day) 
        Rs = shortRad_daily.copy()
        
        
        # (3) Wind speed
        u2m = windspeed_daily.copy()
        # limit to no less than 0.5 m/s; FAO 56, p.63
        u2m[windspeed_daily < 0.5] = 0.5

        # (4) Slope of saturation vapor pressure curve (delta)
        dl = 4098 * (0.6108*np.exp((17.27*ta) / (237.3+ta)))/(ta+237.3)**2
        
        # (5) Atmospheric pressure
        P = 101.3*np.power(((293-(0.0065*alt))/293), 5.26)
        
        # (6) Psychrometric constant
        psy = 0.0016286 * P/lam
        # Aerodynamic resistance 
        ra = 208/u2m
        # Crop canopy reistance
        rc = 100/(0.5*24*0.12)
        # Modified psychrometric constant
        psy_mod = psy*(1+(rc/ra))
        
        
        # (7) Delta Term (DT) - auxiliary calc for Radiation Term
        DT = dl/(dl+psy_mod)*(1/lam)#dl/(dl+psy*(1+0.34*u2m))
        
        # (8) Psi Term (PT) - auxiliary calc for Wind Term
        PT = psy/(dl+psy_mod)  # psy/(dl+psy_mod*(1+0.34*u2m))
        
        # (9) Temperature Term (TT) - auxiliary calc for Wind Term
        TT = u2m * (900/(ta+273))
        
        # (10) Mean Saturation Vapor Pressure derived from air temperature
        es_tmin = 0.6108 * np.exp((17.27 * minT_daily) / (minT_daily + 237.3))
        es_tmax = 0.6108 * np.exp((17.27 * maxT_daily) / (maxT_daily + 237.3))
        es = 0.5*(es_tmin + es_tmax)
        
        # (11) Actual Vapor Pressure derived from relative humidity
        ea = rel_humidity * es # the case for using mean relative humidity
     

        # (12) Inverse relative distance Earth-Sun (dr) and solar declination (sd)
        # Julien Days
        doy = np.arange(cycle_begin, cycle_end+1)
        # Relative distance Earth to Sun
        dr = 1 + 0.033*np.cos(2*np.pi*doy/365)
        # Solar declination [rad]
        sd = 0.4093 * np.sin((2*np.pi*doy/365)-1.405)
        
        # (13) Conversion of latitude in degrees to radians
        lat_rad = latitude * np.pi/180
    
        # (14) Sunset Hour angle
        sha = np.arccos(-np.tan(lat_rad)*np.tan(sd))
        # arccos yields nan for high NH/SH lats in winter
        # we need to replace those nans with 0
        # this fix looks strange because we have to play nice with jit nopython 
        # it works because nan!=nan is True
        for d in doy-1:
            if sha[d]!=sha[d]: sha[d]=0

        # (15) Extraterrestrial radiation (top of atmosphere radiation)
        Ra = 0.082*(24*60/np.pi)*dr*((sha*np.sin(lat_rad)*np.sin(sd))+(np.cos(lat_rad)*np.cos(sd)*np.sin(sha)))

        # (16) Clear-sky solar radiation
        Rso= Ra*(0.75+(0.00002*alt))

        # (17) Net incoming Short-wave radiation
        alpha = 0.23
        Rns = Rs*(1-alpha)

        # (18) Net outgoing long wave-radiation
        sub_cst = 0.000000004903  # Stefan-Boltzmann constant [MJ K-4 m-2 day-1]
        SD_DL = 2*((Rs/Ra)-0.25)
        Rnl = (((273.16+maxT_daily)**4)+((273.16 +minT_daily)**4)/2)*(0.34 - (0.14*(ea**0.5)))*((1.35*(Rs/Rso))-0.35)*sub_cst
        # Rnl = sub_cst*(0.1+0.9*SD_DL)*(0.34 - (0.139*(ea**0.5))) * \
        #     (((273.16+maxT_daily)**4)+((273.16 + minT_daily)**4)/2)
        # (19) Net Radiation flux at crop surface
        Rn = Rns - Rnl
        # Soil Heat Flux - G
        ta_dublicate_last2 = np.append(ta, np.array([ta[-1]]))
        ta_dublicate_first2 = np.append(np.array([ta[-1]]), ta)
        G = 0.14 * (ta_dublicate_last2 - ta_dublicate_first2)
        G = G[0:G.size-1]
        # Rn expressed as the equivalent of evaporation Rng
        Rng = Rn-G #0.408*(Rn-G)

        # FINAL STEPS
        # (FS1) Radiation Term
        ET_rad = DT * Rng
        
        # (FS2) Wind Term
        ET_wind = PT * TT * (es-ea)
        
        # (FS3) Final Reference Evapotranspiration Value
        ET0 = ET_wind + ET_rad
        ET0[ET0 < 0]= 0

        return ET0

    def calculateETO(self):
        return ETOCalc.calculateETONumba(self.cycle_begin, self.cycle_end, self.latitude, self.alt,  self.minT_daily, self.maxT_daily, self.windspeed_daily, self.shortRad_daily, self.rel_humidity)
