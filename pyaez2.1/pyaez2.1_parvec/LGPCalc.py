"""
PyAEZ version 2.1.0 (June 2023)
PyAEZ: LGPCalc.py calculates the length of growing period (LGP)
2022/2023: Kittiphon Boonma 
2024: Kerrie Geil (vectorize and parallelize with dask)

"""

import numpy as np

def rainPeak(meanT_daily, lgpt5):
    """Finds the start and end Julian day of the growing season based
        on daily meanT and a threshold of 5C

    Args:
        meanT_daily (3D float array): daily mean temperature
        lgpt5 (3D float array): thermal length of growing period (T>5C)

    ---
    Returns:
        istart0(2D float array): the starting date of the growing period
        istart1(2D float array): the ending date of the growing period
    """
    # get index of first occurrence in time where true at each grid cell  
    day1=np.argmax(meanT_daily>=5,axis=2)   
    # argmax returns 0 where there is no first occurrence (no growing season) so need to fix  
    day1=np.where(lgpt5==0,np.nan,day1)  
    istart0=np.where((lgpt5<365),day1,0).astype('float32') # replaces if block  
    dat1=np.where(istart0>365,istart0-365,istart0)  # replaces setdat function  
    istart1=np.where((lgpt5<365),dat1+lgpt5-1,lgpt5-1).astype('float32') # replaces if block  

    return istart0, istart1


# func isfromt0 eliminated in vectorization
# func setdat eliminated in vectorization
# func islgpt eliminated in vectorization


def psh(ng, et0):
    """Calculate soil moisture depletion fraction (0-1)

    Args:
        ng (3D float array): crop group (not implemented, is set to 0 in ClimateRegime)
        et0 (3D float array): potential evapotranspiration [mm/day]

    ---
    Returns:
        psh (3D float array): soil moisture depletion fraction
    """
    psh0=np.where(ng==0,0.5,0.3+(ng-1)*.05)
    psh = psh0 + .04 * (5.-et0)
    psh=np.where(psh<0.1,0.1,psh)  
    psh=np.where(psh>0.8,0.8,psh)

    return psh.astype('float32')


def val10day(Et):
    """Calculate forward-looking 10-day moving average 

    Args:
        Et (3D float array): evapotranspiration

    ---
    Returns:
        rollmean3D (3D float array): forward-looking 10-day moving average      
    """
    window_size = 10
    var = np.cumsum(Et, axis=-1, dtype='float32')
    var[:,:,window_size:] = var[:,:,window_size:] - var[:,:,:-window_size]
    rollmean3d = np.round((var[:,:,window_size-1:]/window_size),2)

    # # this is too slow
    # # use numpy convolve to get a forward-looking rolling mean on 1 dimension of a 3D array
    # window_size = 10
    # # first reshape the input data to 2d (space, time)
    # Et2d=np.reshape(Et,(Et.shape[0]*Et.shape[1],Et.shape[2])).astype('float32')
    # # use apply_along_axis to apply a rolling mean with length=window_size
    # rollmean2d=np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size), mode='valid')/window_size, axis=1, arr=Et2d) 
    # # reshape the result back to 3 dimensions
    # rollmean3d=np.reshape(rollmean2d,(Et.shape[0],Et.shape[1],rollmean2d.shape[1])).astype('float32')

    return rollmean3d


def eta(mask,wb_old,etm,Sa,D,p,rain):  

    """SUBROUTINE: Calculate actual evapotranspiration (ETa) 

    Args:
        mask (2D integer array): mask of 0 and 1 where 1 indicates pixels of the 
            same ET regime, shape (im_height,im_width)
        wb_old (2D float array): daily water balance left from the previous day
        etm (2D float array): maximum evapotranspiration for a single day
        Sa (float scalar): Available soil moisture holding capacity [mm/m]
        D (float scalar): rooting depth [m]
        p (2D float array): soil moisture depletion fraction (0-1) for a single day
        rain (2D float array): rainfall for a single day


    Returns:
        wb (2D float array): water balance for a single day
        wx (2D float array): total available soil moisture for a single day
        eta_local(2D float array) the actual evapotranspiration for a single day
    """
    s = wb_old+rain
    wx=np.where(mask==1,np.float32(0),np.float32(np.nan))  
    Salim = max(Sa*D, 1.) 
    wr = np.where(100*(1.-p)>Salim,Salim,100*(1.-p))    
    
    eta_local=np.empty(etm.shape)
    eta_local[:]=np.float32(np.nan)
    eta_local=np.where(rain>=etm,etm,eta_local) 
    eta_local=np.where((~np.isfinite(eta_local))&(s-wr>=etm),etm,eta_local)

    rho=wb_old/wr  
    eta_local=np.where((~np.isfinite(eta_local))&(rain+rho*etm >= etm)&(mask==1),etm,eta_local)  
    eta_local=np.where((~np.isfinite(eta_local))&(rain+rho*etm < etm)&(mask==1),rain+rho*etm,eta_local)  

    wb=s-eta_local          
    wb=np.where(wb>=Salim,Salim,wb)  
    wx=np.where(wb>=Salim,wb-Salim,wx)  
    wx=np.where(wb<Salim,0,wx)            
    wb=np.where(wb<0,0,wb)  

    return wb, wx, eta_local  

# =========================================================

# Here I'm dividing EtaCalc up where each part of the if block in the original EtaCalc is a separate function (I call this EtaClass)
# So EtaClass determines which section of the big if block applies to each pixel
# We send all pixels of each "class" using a mask to the appropriate EtaCalc_xxxx function and then combine the results together 
# I'm doing all this because we generally need to remove if blocks in favor of where statements for vectorization
# and also, we can add parallelization where each part of the if block (EtaCalc_xxxx functions) are computed simultaneously 
# The flow is:
#    ClimateRegime.getLGP first calls LGPCalc.Eta_class (faster outside of any loops)
#    ClimateRegime.getLGP then calls LGPCalc.EtaCalc on a chunk of data inputs 
#    LGPCalc.EtaCalc calls EtaCalc_snow, EtaCalc_snowmelting, EtaCalc_cold, EtaCalc_warm, EtaCalc_warmest for each day and returns only lgp_tot for the data chunk
#    ClimateRegime.getLGP then assembles all resulting lgp_tot chunks into a single array


def Eta_class(mask,lgpt5,ta,tx,tmelt):
    """Determines which set of equations should be used to compute actual evapotranspiration (ETa)

    Class 1 --> Period with Tmax <= Txsnm (precipitaton falls as snow as is added to snow bucket)
    Class 2 --> Snow-melt takes place; minor evapotranspiration
    Class 3 --> Biological activities before start of growing period
    Class 4 --> Growing season
    Class 5 --> I'm just calling this one "warmest"

    Args:
        mask (2D integer array): administrative mask of 0 and 1, if no mask was created by the user this comes in as all 1's, shape (im_height,im_width)
        lgpt5 (2D float array): number of days with mean daily tenmperature above 5 degC, shape (im_height,im_width)
        ta (3D float array): daily average temperature, shape (im_height,im_width,365)
        tx (3D float array): daily maximum temperature, shape (im_height,im_width,365)
        tmelt (float scalar): the maximum temperature threshold, underwhich precip. falls as snow, scalar constant

    Returns:
        eta_class(3D integer array): array of evapotransporation regime classes
    """      
    # this method of assignment uses less RAM than where statements
    eta_class=np.zeros(tx.shape,dtype='int8') # initialization

    # category 1, snow
    inds = (ta<=0)
    inds &= (tx<=tmelt)
    eta_class[inds]=1

    # category 2, snow melting
    inds = (ta<=0)
    inds &= (tx>=0)
    inds &= (eta_class==0)
    eta_class[inds]=2
    
    # category 3, cold pre growing
    inds = (ta>0)
    inds &= (ta<5)
    inds &= (eta_class==0)
    eta_class[inds]=3
    
    # category 4, warm
    inds = (lgpt5<365)
    inds &= (ta>=5)
    inds &= (eta_class==0)
    eta_class[inds]=4
    
    # category 5, warmest
    inds = (mask==1)
    inds &= (eta_class==0)
    eta_class[inds]=5
    
    return eta_class    


# The next 5 functions EtaCalc_xxxxxx are created from sections of the original lengthy EtaCalc IF block

def EtaCalc_snow(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p):
    """compute actual evapotranspiration (ETa) for ET regime "class 1"
       where Tmax <= Txsnm (precipitaton falls as snow as is added to snow bucket)

    Args:
        classmap (2D integer array): the 2D array of eta regime values for a single day (eta_class)
        classnum (integer scalar): the eta regime value (eta_class) that applies to this function (1) for masking 
        kc_list (1D float array): crop coefficients for water requirements
        eto (2D float array): daily value of reference evapotranspiration
        sb_old (2D float array): snow bucket value from the previous day
        pr (2D float array): daily value of precipitation
        wb_old (2D float array): water bucket value from the previous day
        Sa (float scalar): total available soil water holding capacity, set to 100.
        D (float scalar): rooting depth, set to 1.
        p (2D float array): daily share of exess water, below which soil moisture starts to become difficult to extract

    Returns:
        etm (2D float array): daily value of the 'Maximum Evapotranspiration' (mm), shape (im_height,im_width)
        Eta (2D float array): daily value of the 'Actual Evapotranspiration' (mm), shape (im_height,im_width)
        wb (2D float array): daily value of the 'Soil Water Balance', shape (im_height,im_width)
        wx (2D float array): daily value of the 'Maximum water available to plants', shape (im_height,im_width)
        sb (2D float array): daily value of the 'Snow balance' (mm), shape (im_height,im_width)
        kc (2D float array): daily value of the 'crop coefficients for water requirements', shape (im_height,im_width)
        """ 
    # mask everything (only use pixels in this eta regime) 
    mask=np.where(classmap==classnum,1,0)
    eto=np.where(mask==1,eto,np.float32(np.nan))
    sb_old=np.where(mask==1,sb_old,np.float32(np.nan))
    pr=np.where(mask==1,pr,np.float32(np.nan))
    wb_old=np.where(mask==1,wb_old,np.float32(np.nan))
    p=np.where(mask==1,p,np.float32(np.nan))

    kc=np.zeros(mask.shape,dtype='float32')   
    kc=np.where(mask==1,kc_list[0],kc)   
    etm = kc * eto
    del eto, kc
    sbx = sb_old + pr 
    # snm=np.zeros(sb_old.shape,dtype='float32') # for debugging
    del sb_old    

    # call the eta subroutine
    wb, wx, Eta = eta(mask,wb_old-pr, etm, Sa, D, p, pr)
    del mask,pr,p,wx

    Salim=Sa*D
    sb=np.where(sbx>=etm,sbx-etm,0)
    Eta=np.where(sbx>=etm,etm,Eta)

    wb=np.where(sbx>=etm,wb_old-etm,wb)
    wb=np.where((sbx>=etm) & (wb>Salim),Salim,wb)  
    wb=np.where(wb<0,0,wb)  

    # what are we using wx for
    # wx=np.where((sbx>=etm) & (wb>Salim),wb-Salim,wx)
    # wx=np.where((sbx>=etm) & (wb<=Salim),0,wx) 

    return [etm, Eta, wb, sb]


def EtaCalc_snowmelting(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):
    """compute actual evapotranspiration (ETa) for ET regime "class 2"
       where Snow-melt takes place; minor evapotranspiration

    Args:
        classmap (2D integer array): the 2D array of eta regime values for a single day (eta_class)
        classnum (integer scalar): the eta regime value (eta_class) that applies to this function (2) for masking 
        kc_list (1D float array): crop coefficients for water requirements
        eto (2D float array): daily value of reference evapotranspiration
        sb_old (2D float array): snow bucket value from the previous day
        pr (2D float array): daily value of precipitation
        wb_old (2D float array): water bucket value from the previous day
        Sa (float scalar): total available soil water holding capacity, set to 100.
        D (float scalar): rooting depth, set to 1.
        p (2D float array): daily share of exess water, below which soil moisture starts to become difficult to extract
        Fsnm (float scalar): snow melt parameter, set to 5.5        
        tx (2D float array): daily value of maximum temperature
        tmelt (float scalar): the maximum temperature threshold underwhich precip falls as snow, set to 0

    Returns:
        etm (2D float array): daily value of the 'Maximum Evapotranspiration' (mm), shape (im_height,im_width)
        Eta (2D float array): daily value of the 'Actual Evapotranspiration' (mm), shape (im_height,im_width)
        wb (2D float array): daily value of the 'Soil Water Balance', shape (im_height,im_width)
        wx (2D float array): daily value of the 'Maximum water available to plants', shape (im_height,im_width)
        sb (2D float array): daily value of the 'Snow balance' (mm), shape (im_height,im_width)
        kc (2D float array): daily value of the 'crop coefficients for water requirements', shape (im_height,im_width)
        """
    # mask everything (only use pixels in this eta regime) 
    mask=np.where(classmap==classnum,1,0)
    eto=np.where(mask==1,eto,np.float32(np.nan))
    sb_old=np.where(mask==1,sb_old,np.float32(np.nan))
    pr=np.where(mask==1,pr,np.float32(np.nan))
    wb_old=np.where(mask==1,wb_old,np.float32(np.nan))
    p=np.where(mask==1,p,np.float32(np.nan))
    tx=np.where(mask==1,tx,np.float32(np.nan))

    kc=np.zeros(mask.shape,dtype='float32')   
    kc=np.where(mask==1,kc_list[1],kc)
    etm = kc * eto
    del eto, kc

    snm=np.zeros(sb_old.shape,dtype='float32')
    snm=np.where((sb_old>0.) & (Fsnm*(tx-tmelt)>sb_old),sb_old, snm) 
    snm=np.where((sb_old>0.) & (Fsnm*(tx-tmelt)<sb_old),Fsnm*(tx-tmelt),snm)     
    del tx
    sbx=sb_old-snm 
    del sb_old
    Salim = Sa*D

    sb=np.where(sbx>=etm,sbx-etm,0)

    # call the eta subroutine
    wb, wx, Eta = eta(mask,wb_old+snm, etm, Sa, D, p, pr)
    del mask, p, wx

    Eta=np.where(sbx>=etm,etm,Eta)
    wb=np.where(sbx>=etm,wb_old+snm+pr-etm,wb)
    del pr,snm 
    wb=np.where((sbx>=etm) & (wb>Salim),Salim,wb)

    # what are we using wx for
    # wx=np.where((sbx>=etm) & (wb>Salim),wb-Salim,wx)
    # wx=np.where((sbx>=etm) & (wb<=Salim),0,wx)

    wb=np.where(wb<0,0,wb)

    return [etm, Eta, wb, sb]

def EtaCalc_cold(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):
    """compute actual evapotranspiration (ETa) for ET regime "class 3"
       where there are Biological activities before start of growing period

    Args:
        classmap (2D integer array): the 2D array of eta regime values for a single day (eta_class)
        classnum (integer scalar): the eta regime value (eta_class) that applies to this function (3) for masking 
        kc_list (1D float array): crop coefficients for water requirements
        eto (2D float array): daily value of reference evapotranspiration
        sb_old (2D float array): snow bucket value from the previous day
        pr (2D float array): daily value of precipitation
        wb_old (2D float array): water bucket value from the previous day
        Sa (float scalar): total available soil water holding capacity, set to 100.
        D (float scalar): rooting depth, set to 1.
        p (2D float array): daily share of exess water, below which soil moisture starts to become difficult to extract
        Fsnm (float scalar): snow melt parameter, set to 5.5        
        tx (2D float array): daily value of maximum temperature
        tmelt (float scalar): the maximum temperature threshold underwhich precip falls as snow, set to 0

    Returns:
        etm (2D float array): daily value of the 'Maximum Evapotranspiration' (mm), shape (im_height,im_width)
        Eta (2D float array): daily value of the 'Actual Evapotranspiration' (mm), shape (im_height,im_width)
        wb (2D float array): daily value of the 'Soil Water Balance', shape (im_height,im_width)
        wx (2D float array): daily value of the 'Maximum water available to plants', shape (im_height,im_width)
        sb (2D float array): daily value of the 'Snow balance' (mm), shape (im_height,im_width)
        kc (2D float array): daily value of the 'crop coefficients for water requirements', shape (im_height,im_width)
        """
    # mask everything (only use pixels in this eta regime) 
    mask=np.where(classmap==classnum,1,0)
    eto=np.where(mask==1,eto,np.float32(np.nan))
    sb_old=np.where(mask==1,sb_old,np.float32(np.nan))
    pr=np.where(mask==1,pr,np.float32(np.nan))
    wb_old=np.where(mask==1,wb_old,np.float32(np.nan))
    p=np.where(mask==1,p,np.float32(np.nan))
    tx=np.where(mask==1,tx,np.float32(np.nan))

    kc=np.zeros(mask.shape,dtype='float32')   
    kc=np.where(mask==1,kc_list[2],kc)
    etm = kc * eto
    del eto, kc

    snm=np.zeros(sb_old.shape,dtype='float32')
    snm=np.where((sb_old>0.) & (Fsnm*(tx-tmelt)>sb_old),sb_old, snm) 
    snm=np.where((sb_old>0.) & (Fsnm*(tx-tmelt)<sb_old),Fsnm*(tx-tmelt),snm)   
    snm = np.where(sb_old<=0, 0, snm)   
    sb=sb_old-snm 
    del tx,sb_old

    # call the eta subroutine
    wb, wx, Eta = eta(mask,wb_old+snm, etm, Sa, D, p, pr)
    del mask, wb_old, p, pr, wx, snm

    Eta=np.where(Eta>etm,etm,Eta)

    return [etm, Eta, wb, sb]


def EtaCalc_warm(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt,istart,iend,idoy):
    """compute actual evapotranspiration (ETa) for ET regime "class 4" during the growing season

    Args:
        classmap (2D integer array): the 2D array of eta regime values for a single day (eta_class)
        classnum (integer scalar): the eta regime value (eta_class) that applies to this function (4) for masking 
        kc_list (1D float array): crop coefficients for water requirements
        eto (2D float array): daily value of reference evapotranspiration
        sb_old (2D float array): snow bucket value from the previous day
        pr (2D float array): daily value of precipitation
        wb_old (2D float array): water bucket value from the previous day
        Sa (float scalar): total available soil water holding capacity, set to 100.
        D (float scalar): rooting depth, set to 1.
        p (2D float array): daily share of exess water, below which soil moisture starts to become difficult to extract
        Fsnm (float scalar): snow melt parameter, set to 5.5        
        tx (2D float array): daily value of maximum temperature
        tmelt (float scalar): the maximum temperature threshold underwhich precip falls as snow, set to 0
        istart (2D float array): starting julian day of the growing period
        iend (2D float array): ending julian day of the growing period
        idoy (integer scalar): index of the time loop
    Returns:
        etm (2D float array): daily value of the 'Maximum Evapotranspiration' (mm), shape (im_height,im_width)
        Eta (2D float array): daily value of the 'Actual Evapotranspiration' (mm), shape (im_height,im_width)
        wb (2D float array): daily value of the 'Soil Water Balance', shape (im_height,im_width)
        wx (2D float array): daily value of the 'Maximum water available to plants', shape (im_height,im_width)
        sb (2D float array): daily value of the 'Snow balance' (mm), shape (im_height,im_width)
        kc (2D float array): daily value of the 'crop coefficients for water requirements', shape (im_height,im_width)
        """    
    # mask everything (only use pixels in this eta regime) 
    mask=np.where(classmap==classnum,1,0)
    eto=np.where(mask==1,eto,np.float32(np.nan))
    sb_old=np.where(mask==1,sb_old,np.float32(np.nan))
    pr=np.where(mask==1,pr,np.float32(np.nan))
    wb_old=np.where(mask==1,wb_old,np.float32(np.nan))
    p=np.where(mask==1,p,np.float32(np.nan))
    tx=np.where(mask==1,tx,np.float32(np.nan))

    kc3=np.zeros(mask.shape,dtype='float32')   
    kc3=np.where(mask==1,kc_list[3],kc3)
    kc4=np.zeros(mask.shape,dtype='float32')   
    kc4=np.where(mask==1,kc_list[4],kc4)
    xx=np.where((mask==1) & (idoy-istart>=0) & (idoy-iend<=0) & ((idoy-istart)/30.>1.),1., (idoy-istart)/30.)
    kc=np.where((idoy-istart>=0) & (idoy-iend<=0), kc3*(1.-xx)+(kc4*xx), kc3)
    del kc3,kc4,xx

    etm = kc * eto
    del eto, kc

    snm=np.zeros(sb_old.shape,dtype='float32')
    snm=np.where((sb_old>0.) & (Fsnm*(tx-tmelt)>sb_old),sb_old, snm) 
    snm=np.where((sb_old>0.) & (Fsnm*(tx-tmelt)<sb_old),Fsnm*(tx-tmelt),snm) 
    del tx
    sb=sb_old-snm
    del sb_old

    # call the eta subroutine
    wb, wx, Eta = eta(mask,wb_old+snm, etm, Sa, D, p, pr)
    del mask,wb_old,p,pr,wx,snm
    Eta=np.where(Eta>etm,etm,Eta)

    return [etm, Eta, wb, sb]


def EtaCalc_warmest(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):
    """compute actual evapotranspiration (ETa) for ET regime "class 5" the warmest period

    Args:
        classmap (2D integer array): the 2D array of eta regime values for a single day (eta_class)
        classnum (integer scalar): the eta regime value (eta_class) that applies to this function (4) for masking 
        kc_list (1D float array): crop coefficients for water requirements
        eto (2D float array): daily value of reference evapotranspiration
        sb_old (2D float array): snow bucket value from the previous day
        pr (2D float array): daily value of precipitation
        wb_old (2D float array): water bucket value from the previous day
        Sa (float scalar): total available soil water holding capacity, set to 100.
        D (float scalar): rooting depth, set to 1.
        p (2D float array): daily share of exess water, below which soil moisture starts to become difficult to extract
        Fsnm (float scalar): snow melt parameter, set to 5.5        
        tx (2D float array): daily value of maximum temperature
        tmelt (float scalar): the maximum temperature threshold underwhich precip falls as snow, set to 0

    Returns:
        etm (2D float array): daily value of the 'Maximum Evapotranspiration' (mm), shape (im_height,im_width)
        Eta (2D float array): daily value of the 'Actual Evapotranspiration' (mm), shape (im_height,im_width)
        wb (2D float array): daily value of the 'Soil Water Balance', shape (im_height,im_width)
        wx (2D float array): daily value of the 'Maximum water available to plants', shape (im_height,im_width)
        sb (2D float array): daily value of the 'Snow balance' (mm), shape (im_height,im_width)
        kc (2D float array): daily value of the 'crop coefficients for water requirements', shape (im_height,im_width)
        """
    # mask everything (only use pixels in this eta regime) 
    mask=np.where(classmap==classnum,1,0)
    eto=np.where(mask==1,eto,np.float32(np.nan))
    sb_old=np.where(mask==1,sb_old,np.float32(np.nan))
    pr=np.where(mask==1,pr,np.float32(np.nan))
    wb_old=np.where(mask==1,wb_old,np.float32(np.nan))
    p=np.where(mask==1,p,np.float32(np.nan))
    tx=np.where(mask==1,tx,np.float32(np.nan))

    kc=np.zeros(mask.shape,dtype='float32')   
    kc=np.where(mask==1,kc_list[4],kc) 
    etm = kc * eto
    del eto,kc

    snm=np.zeros(sb_old.shape,dtype='float32')
    snm=np.where((sb_old>0.) & (Fsnm*(tx-tmelt)>sb_old),sb_old, snm) 
    snm=np.where((sb_old>0.) & (Fsnm*(tx-tmelt)<sb_old),Fsnm*(tx-tmelt),snm)     
    del tx
    sb=sb_old-snm #???? repo version doesn't have this line, which means it is not computing sb correctly for this class
    del sb_old

    # call the eta subroutine
    wb, wx, Eta = eta(mask,wb_old, etm, Sa, D, p, pr)
    del mask,wb_old,p,pr,wx

    Eta=np.where(Eta>etm,etm,Eta)

    return [etm, Eta, wb, sb]


def EtaCalc(im_mask,Tx,islgp,Pr,Txsnm,Fsnm,Eto,wb_old,sb_old,istart0,istart1,Sa,D,p,kc_list,lgpt5,eta_class,doy_start,doy_end,parallel):

    """vectorized calculation of actual evapotranspiration (ETa)
    
    Args:
        im_mask (2D integer array): administrative mask of 0 and 1, if no mask was created by the user this comes in as all 1's, shape (im_height,im_width)
        Tx (3D float array): daily maximum temperature
        islgp (3D integer array): array of 0 & 1 where 1 indicates days where mean temperature is >=5C
        Pr (3D float array): daily precipitation
        Txsnm (float scalar): maximum temperature threshold underwhich precip falls as snow, set to 0
        Fsnm (float, scalar): snow melt parameter, set to 5.5                
        Eto (3D float array): daily reference evapotranspiration
        wb_old (2D float array): water bucket initialized to zero
        sb_old (2D float array): snow bucket initialized to zero        
        istart0 (2D float array): starting julian day of the growing period
        istart1 (2D float array): ending julian day of the growing period
        Sa (float scalar): total available soil water holding capacity, set to 100.
        D (float scalar): rooting depth, set to 1.
        p (2D float array): daily share of exess water, below which soil moisture starts to become difficult to extract
        kc_list (1D float array): crop coefficients for water requirements        
        lgpt5 (float): number of days with mean daily temperature above 5 degC
        eta_class (3D integer array): eta regime values (determines which EtaCalc_xxxxx function applies)
        doy_start (integer scalar): data julian start day, defaults to 1
        doy_end (integer scalar): data julian end day, defaults to 365
        parallel (boolean): flag determining whether to run in parallel

    Returns:
        lgp_tot ()      
    """  
    if parallel:
        import dask

    ETM_list=[]  # list to hold 365 arrays (results for each day)
    ETA_list=[]  # list to hold 365 arrays (results for each day)
    # WB_list=[]   # list to hold 365 arrays (results for each day) # for debugging
    # SB_list=[]   # list to hold 365 arrays (results for each day) # for debuggin


    for idoy in range(doy_start-1, doy_end):

        # compute etm,eta,wb,sb for each eta regime in parallel each day
        if parallel:      
            # we parallelize here by eta_class (the five different regimes for computing ET)
            # the computations for eta_class are each a delayed function
            # we call each function which saves the future computation as an object to a list of tasks
            # then we call compute on the list of tasks to execute them in parallel 

            results_list=[]  # list of lists: [[etm, eta, wb, sb],[etm, eta, wb, sb],...] in that order
        
            # delay data inputs so they're passed only once to the computations
            eta_class_d=dask.delayed(eta_class[:,:,idoy])
            Eto_d=dask.delayed(Eto[:,:,idoy])
            sb_old_d=dask.delayed(sb_old)
            Pr_d=dask.delayed(Pr[:,:,idoy])
            wb_old_d=dask.delayed(wb_old)
            p_d=dask.delayed(p[:,:,idoy])
            Tx_d=dask.delayed(Tx[:,:,idoy])

            # dask.delayed call to each ETACalc_xxxxx
            # vars1, vars2... are each a list of arrays etm,eta,wb,sb in that order
            vars1=dask.delayed(EtaCalc_snow)(eta_class_d,1,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d)
            vars2=dask.delayed(EtaCalc_snowmelting)(eta_class_d,2,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d,Fsnm,Tx_d,Txsnm)                                                                       
            vars3=dask.delayed(EtaCalc_cold)(eta_class_d,3,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d,Fsnm,Tx_d,Txsnm)                                                               
            vars4=dask.delayed(EtaCalc_warm)(eta_class_d,4,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d,Fsnm,Tx_d,Txsnm,istart0,istart1,idoy)                                        
            vars5=dask.delayed(EtaCalc_warmest)(eta_class_d,5,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d,Fsnm,Tx_d,Txsnm)
            task_list=[vars1,vars2,vars3,vars4,vars5] # list of lists

            # compute in parallel
            results_list=dask.compute(*task_list)

            # aggregate results (spatially for this idoy) from the 5 different ETACalc routines for each variable
            arr_shape=results_list[0][0].shape
            ETM_agg=np.empty(arr_shape,dtype='float32')
            ETA_agg=np.empty(arr_shape,dtype='float32')
            WB_agg=np.empty(arr_shape,dtype='float32')
            SB_agg=np.empty(arr_shape,dtype='float32')
            ETM_agg[:],ETA_agg[:],WB_agg[:],SB_agg[:]=np.float32(np.nan),np.float32(np.nan),np.float32(np.nan),np.float32(np.nan)
            for icat,results in enumerate(results_list):
                ETM_agg=np.where(eta_class[:,:,idoy]==icat+1,results[0],ETM_agg)
                ETA_agg=np.where(eta_class[:,:,idoy]==icat+1,results[1],ETA_agg)
                WB_agg=np.where(eta_class[:,:,idoy]==icat+1,results[2],WB_agg)
                SB_agg=np.where(eta_class[:,:,idoy]==icat+1,results[3],SB_agg)

        else:
            ### IF parallel=False COMPUTE WITHOUT DASK ###    
            results_list=[] # list of lists: [[etm, eta, wb, sb],[etm, eta, wb, sb],...] in that order

            # call to each ETACalc_xxxxx    
            # vars1, vars2... are each a list of arrays etm,eta,wb,sb in that order
            vars1=EtaCalc_snow(eta_class[:,:,idoy],1,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy])
            vars2=EtaCalc_snowmelting(eta_class[:,:,idoy],2,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm)                                                                       
            vars3=EtaCalc_cold(eta_class[:,:,idoy],3,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm)                                                               
            vars4=EtaCalc_warm(eta_class[:,:,idoy],4,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm,istart0,istart1,idoy)                                        
            vars5=EtaCalc_warmest(eta_class[:,:,idoy],5,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm)
            results_list.append([vars1,vars2,vars3,vars4,vars5]) # list of lists           

            # aggregate results from the 5 different ETACalc routines for each variable
            arr_shape=results_list[0][0][0].shape
            ETM_agg=np.empty(arr_shape,dtype='float32')
            ETA_agg=np.empty(arr_shape,dtype='float32')
            WB_agg=np.empty(arr_shape,dtype='float32')
            SB_agg=np.empty(arr_shape,dtype='float32')
            ETM_agg[:],ETA_agg[:],WB_agg[:],SB_agg[:]=np.float32(np.nan),np.float32(np.nan),np.float32(np.nan),np.float32(np.nan)
            for icat,results in enumerate(results_list[0]):
                ETM_agg=np.where(eta_class[:,:,idoy]==icat+1,results[0],ETM_agg)
                ETA_agg=np.where(eta_class[:,:,idoy]==icat+1,results[1],ETA_agg)
                WB_agg=np.where(eta_class[:,:,idoy]==icat+1,results[2],WB_agg)
                SB_agg=np.where(eta_class[:,:,idoy]==icat+1,results[3],SB_agg)
        
        # collect the aggregated results for each day in a list
        ETM_list.append(ETM_agg)
        ETA_list.append(ETA_agg)
        # WB_list.append(WB_agg) # for debugging
        # SB_list.append(SB_agg) # for debugging 

        # update wb and sb values
        wb_old=WB_agg.copy()
        sb_old=SB_agg.copy()

        if parallel:
            del eta_class_d, Eto_d, sb_old_d, Pr_d, wb_old_d, p_d, Tx_d  # clean up
        
        # clean up
        del vars1, vars2, vars3, vars4, vars5, task_list, results_list
        del ETM_agg, ETA_agg, WB_agg, SB_agg

        # the daily loop ends here
    
    # more clean up to release RAM
    del im_mask,Txsnm,Fsnm,Sa,D,kc_list
    del lgpt5,eta_class,istart0,istart1
    del Eto, sb_old, Pr, wb_old, p, Tx

    # create single array (all days together) for ETM, ETA
    ETM=np.stack(ETM_list,axis=-1,dtype='float32') # stack on last dim (time)
    ETA=np.stack(ETA_list,axis=-1,dtype='float32') # stack on last dim (time)
    # WB=np.stack(WB_list,axis=-1,dtype='float32') # stack on last dim (time) # for debugging
    # SB=np.stack(SB_list,axis=-1,dtype='float32') # stack on last dim (time) # for debugging
    del ETM_list,ETA_list

    # eliminate negatives
    ETA=np.where(ETA<0,0,ETA)

    # extend ETM,ETA so that the 10day rolling mean in
    # the next step doesn't reduce the time dimension
    with np.errstate(invalid='ignore'):  
        ETMx = np.append(ETM, ETM[:,:,0:12],axis=2)  
        ETAx = np.append(ETA, ETA[:,:,0:12],axis=2)  
    del ETM, ETA

    # compute forward-looking 10day rolling mean
    xx = val10day(ETAx)  
    yy = val10day(ETMx)  
    del ETMx,ETAx

    # compute growing season length
    with np.errstate(divide='ignore', invalid='ignore'):  
        lgp_whole = xx[:,:,:doy_end]/yy[:,:,:doy_end]  
    del xx,yy

    lgp_tot=np.where((islgp==1)&(lgp_whole>=0.4),1,0).sum(axis=2)  
    del lgp_whole   
 
    return lgp_tot  


