"""
PyAEZ version 2.1.0 (June 2023)
ETOCalc.py calculates the reference evapotranspiration from 
the climatic data provided by the PyAEZ user.
2020: N. Lakmal Deshapriya, Thaileng Thol
2022/2023: Kittiphon Boonma  (Numba)

"""
import numpy as np

class ETOCalc(object):
    # __init__ and setClimateData functions removed to save RAM

    def calculateETO(self,dstart,dend,lat,alt,tmn,tmx,u2m,srad,rh):    
        # numba doesn't speed this up in time tests 
        # removing in favor of vectorization which also allows chunking with dask for speed 
        # this calculation has been reordered to decrease memory use
        """Calculate the reference evapotranspiration with Penmann-Monteith Equation

        Returns:
            ## float: ETo of a single pixel (function is called pixel-wise)
            float: ETo of each pixel    
        """   
        nlats=tmn.shape[0]
        nlons=tmn.shape[1]
        dayoyr = np.arange(dstart, dend+1)  # Julien Days   
        ndays=len(dayoyr)            

        # calculate extraterrestrial radiation 
        #-------------------------------------
        latr = (lat * np.pi/180.).astype('float32')
        latr = np.broadcast_to(latr[:,:,np.newaxis],(nlats,nlons,ndays))

        # solar declination (rad)
        sdcl = (0.4093 * np.sin(0.017214206 * dayoyr - 1.405)).astype('float32')
        sdcl = np.broadcast_to(sdcl[np.newaxis, np.newaxis,:],(nlats,nlons,ndays))

        # sunset hour angle
        # top=(-np.tan(latr)*np.tan(sdcl))
        # X = 1 - (np.tan(latr)**2)*(np.tan(sdcl)**2)
        # inds=(X<=0)
        # X[inds]=1.E-5
        # omg = 1.5708 - np.arctan(top/(X**0.5))
        # omg[inds]=0.
        # del top,X,inds      

        # relative distance earth to sun
        sdst = (1.0 + 0.033 * np.cos(0.017214206 * dayoyr)).astype('float32')
        sdst = np.broadcast_to(sdst[np.newaxis, np.newaxis,:],(nlats,nlons,ndays))
        del dayoyr

        # extraterrestrial radiation
        xx = (np.sin(sdcl) * np.sin(latr))
        yy = (np.cos(sdcl) * np.cos(latr))
        del sdcl,latr

        zz=xx/yy
        omg = np.tan(zz / (1. - zz*zz)**0.5) + 1.5708
        omg=np.where((np.abs(zz) >= 0.9999)&(zz > 0),np.float32(np.pi),omg)
        omg=np.where((np.abs(zz) >= 0.9999)&(zz <= 0),0,omg)
        del zz


        ra = np.round((37.586 * sdst * (omg*xx + np.sin(omg)*yy)).astype('float32'),1)
        # print('omg,sdst,ra',omg.dtype,sdst.dtype,ra.dtype)
        del sdst, omg, xx, yy

        # Mean Saturation Vapor Pressure derived from air temperature
        #-------------------------------------
        es_tmin = 0.6108 * np.exp((17.27 * tmn) / (tmn + 237.3))
        es_tmax = 0.6108 * np.exp((17.27 * tmx) / (tmx + 237.3))
        es = 0.5*(es_tmin + es_tmax)
        ea = rh * es  # Actual Vapor Pressure derived from relative humidity
        del rh
        # print('es_tmin,es_tmax,es,ea',es_tmin.dtype,es_tmax.dtype,es.dtype,ea.dtype)

        # slope vapour pressure curve
        #-------------------------------------
        dlmx = 4098. * es_tmax / (tmx + 237.3)**2
        del es_tmax
        dlmn = 4098. * es_tmin / (tmn + 237.3)**2
        del es_tmin
        dl = 0.5* (dlmx + dlmn)
        # print('dlmx,dlmn,dl',dlmx.dtype,dlmn.dtype,dl.dtype)
        del dlmx,dlmn    

        # solar radiation Rs (0.25, 0.50 Angstrom coefficients)
        #-------------------------------------
        # rs = (0.25 + (0.50 * (sd/dayhr))) * ra
        alt = np.broadcast_to(alt[:,:,np.newaxis],(nlats,nlons,ndays))
        rs0 = (0.75 + 0.00002 * alt) * ra
        # print('alt,rs0',alt.dtype,rs0.dtype)
      
        # net longwave radiation Rnl
        #-------------------------------------
        sub_cst = 4.903E-9 # Stefan-Boltzmann constant [MJ K-4 m-2 day-1]
        with np.errstate(invalid='ignore',divide='ignore'):
            rs_div_ds0=(srad/rs0)
        del rs0
        rnl = (((273.16+tmx)**4)+((273.16 + tmn)**4)) * \
            (0.34 - (0.14*(ea**0.5))) * \
            ((1.35*(rs_div_ds0))-0.35)*sub_cst/2
        del rs_div_ds0

        # net shortwave radiation Rns = (1 - alpha) * Rs
        # (alpha for grass = 0.23)
        #-------------------------------------
        rns = 0.77 * srad
        del srad
        # (g) net radiation Rn = Rns - Rnl
        rn = rns - rnl
        # print('rnl,rns,rn',rnl.dtype,rns.dtype,rn.dtype)
        del rns,rnl

        # soil heat flux [MJ/m2/day]
        #-------------------------------------
        tavg = 0.5*(tmn + tmx)  # Averaged temperature    
        del tmn,tmx
        ta_diff=np.diff(tavg,n=1,axis=2,prepend=tavg[:,:,-1:])    
        G = 0.14 * ta_diff    
        # print('tavg,ta_diff,G',tavg.dtype,ta_diff.dtype,G.dtype)
        del ta_diff

        #-------------------------------------        
        # Psychrometric constant
        lam = 2.501 - 0.002361 * tavg  # Latent heat of vaporization

        # Atmospheric pressure
        ap = 101.3*np.power(((293-(0.0065*alt))/293), 5.256)
        del alt

        gam = 0.0016286 * ap/lam
        # print('lam,ap,gam',lam.dtype,ap.dtype,gam.dtype)
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
        # print('rhoa,gamst',rhoa.dtype,gamst.dtype)
        del rhoa
        #-------------------------------------


        # calculate aerodynamic and radiation terms of ET0
        #-------------------------------------
        et0ady = gam/(dl+gamst) * 900./(tavg+273.) * u2m * (es-ea)
        del gam,tavg,u2m,es,ea

        et0rad = dl/(dl+gamst) * (rn-G) / lam
        del dl,gamst,rn,G,lam

        # calculate ET0
        et0 = et0ady + et0rad
        # print('et0ady,et0rad,et0',et0ady.dtype,et0rad.dtype,et0.dtype)
        del et0ady,et0rad

        et0 = np.where(et0<0,0,et0)
        # print('et0',et0.dtype)

        return et0.astype('float32'), et0ady,et0rad
        # return gamst