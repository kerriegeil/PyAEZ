"""
PyAEZ version 2.1.0 (June 2023)
PyAEZ: LGPCalc.py calculates the length of growing period (LGP)
2022/2023: Kittiphon Boonma 
"""

import numpy as np
# from time import time as timer


def rainPeak(meanT_daily, lgpt5):
    """Scan the monthly precipitation for the month with peak rainfall over 3 months

    Args:
        totalPrec_monthly (float): value of monthly precipitation for a pixel
        meanT_daily (float): daily mean temperature for a pixel
        lgpt5_point (float): thermal length of growing period (T>5C) for a pixel

    Returns:
        meanT_daily(1D NumPy): the smoothened daily mean temperature curve
        istart0(int): the starting date of the growing period
        istart1(int): the ending date of the growing period
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
        ng (float): crop group
        et0 (float): potential evapotranspiration [mm/day]

    Returns:
        float: soil moisture depletion fraction
    """
    psh0=np.where(ng==0,0.5,0.3+(ng-1)*.05)
    psh = psh0 + .04 * (5.-et0)
    psh=np.where(psh<0.1,0.1,psh)  
    psh=np.where(psh>0.8,0.8,psh)

    return psh.astype('float32')


def val10day(Et):
    """Calculate forward-looking 10-day moving average 

    Args:
        
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
        mask (integer): mask of 0 and 1 where 1 indicates pixels of the same ET regime, shape (im_height,im_width)
        wb_old (float): daily water balance left from the previous day
        etm (float): maximum evapotranspiration
        Sa (float): Available soil moisture holding capacity [mm/m]
        D (float): rooting depth [m]
        p (float): soil moisture depletion fraction (0-1)
        rain (float): amount of rainfall


    Returns:
        float: a value for daily water balance
        float: a value for total available soil moisture
        float: the calculated actual evapotranspiration
    """
    s = wb_old+rain
    wx=np.where(mask==1,np.float32(0),np.float32(np.nan))  
    Salim = max(Sa*D, 1.) 
    wr = np.where(100*(1.-p)>Salim,Salim,100*(1.-p))    
    # print('wx,wr',wx.dtype,wr.dtype)
    
    eta_local=np.empty(etm.shape)
    eta_local[:]=np.float32(np.nan)

    eta_local=np.where(rain>=etm,etm,eta_local) 
    # eta_local=np.where(rain>=etm,np.float32(etm),np.float32(np.nan))  
    # print('eta_local1',eta_local.dtype)
    eta_local=np.where((~np.isfinite(eta_local))&(s-wr>=etm),etm,eta_local)
    # eta_local=np.where((s-wr>=etm)& ~np.isfinite(eta_local),np.float32(etm),np.float32(eta_local))  
    # print('eta_local2',eta_local.dtype)
    rho=wb_old/wr  
    # print('rho',rho.dtype)
    eta_local=np.where((~np.isfinite(eta_local))&(rain+rho*etm >= etm)&(mask==1),etm,eta_local)  
    # eta_local=np.where((rain+rho*etm >=etm) & (mask==1) & ~np.isfinite(eta_local),np.float32(etm),np.float32(eta_local))  
    # print('eta_local3',eta_local.dtype)
    eta_local=np.where((~np.isfinite(eta_local))&(rain+rho*etm < etm)&(mask==1),rain+rho*etm,eta_local)  
    # eta_local=np.where((rain+rho*etm <etm) & (mask==1) & ~np.isfinite(eta_local),np.float32(rain+rho*etm),np.float32(eta_local))  
    # print('eta_local4',eta_local.dtype)

    wb=s-eta_local          
    wb=np.where(wb>=Salim,Salim,wb)  
    # print('wb2',wb.dtype)
    wx=np.where(wb>=Salim,wb-Salim,wx)  
    # print('wx2',wx.dtype)
    wx=np.where(wb<Salim,0,wx)            
    # print('wx3',wx.dtype)
    wb=np.where(wb<0,0,wb)  
    # print('wb4',wb.dtype)

    return wb, wx, eta_local  

# =========================================================

# Here I'm dividing EtaCalc up where each part of the if block in the original EtaCalc is a separate function (I call this EtaClass)
# So EtaClass determines which section of the big if block applies to each pixel
# We send all pixels of each "class" using a mask to the appropriate EtaCalc_xxxx function and then combine the results together 
# I'm doing all this because we generally need to remove if blocks and replace with where statements for vectorization
# and also, we can add parallelization where each part of the if block (EtaCalc_xxxx functions) are computed simultaneously 
# The flow is:
#    ClimateRegime.getLGP first calls LGPCalc.Eta_class (faster outside of any loops)
#    ClimateRegime.getLGP then calls LGPCalc.EtaCalc for each chunk of data inputs 
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
        mask (integer): administrative mask of 0 and 1, if no mask was created by the user this comes in as all 1's, shape (im_height,im_width)
        lgpt5 (float): number of days with mean daily tenmperature above 5 degC, shape (im_height,im_width)
        ta (float): daily average temperature, shape (im_height,im_width,365)
        tx (float): daily maximum temperature, shape (im_height,im_width,365)
        tmelt (float): the maximum temperature threshold, underwhich precip. falls as snow, scalar constant

    Returns:
        float: array of evapotransporation regime classes, shape (im_height,im_width)
    """      
    # this method of assignment uses less RAM than where statements
    eta_class=np.zeros(tx.shape,dtype='int8')
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


# The next 5 functions EtaCalc_xxxxxx are created from sections of the original lengthy IF block

def EtaCalc_snow(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p):
    """compute actual evapotranspiration (ETa) for ET regime "class 1"
       where Tmax <= Txsnm (precipitaton falls as snow as is added to snow bucket)

    Args:
        mask (integer): mask of 0 and 1 where 1 indicates pixels in this ET regime, shape (im_height,im_width)
        kc_list (list): crop coefficients for water requirements
        eto (float): daily value of reference evapotranspiration
        sb_old (float): snow bucket value from the previous day
        pr (float): daily value of precipitation
        wb_old (float): water bucket value from the previous day
        Sa (int): total available soil water holding capacity
        D (int): rooting depth
        p (int): the share of exess water, below which soil moisture starts to become difficult to extract

    Returns:
        etm (float): daily value of the 'Maximum Evapotranspiration' (mm), shape (im_height,im_width)
        Eta (float): daily value of the 'Actual Evapotranspiration' (mm), shape (im_height,im_width)
        wb (float): daily value of the 'Soil Water Balance', shape (im_height,im_width)
        wx (float): daily value of the 'Maximum water available to plants', shape (im_height,im_width)
        sb (float): daily value of the 'Snow balance' (mm), shape (im_height,im_width)
        kc (float): daily value of the 'crop coefficients for water requirements', shape (im_height,im_width)
        """ 
    # mask everything 
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
    # snm=np.zeros(sb_old.shape,dtype='float32')
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

    # print('snow',mask.dtype,eto.dtype,sb_old.dtype,pr.dtype,wb_old.dtype,p.dtype,kc.dtype,sbx.dtype,etm.dtype,Eta.dtype,wb.dtype,sb.dtype)
    return [etm, Eta, wb, sb]#,snm]


def EtaCalc_snowmelting(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):
    """compute actual evapotranspiration (ETa) for ET regime "class 2"
       where Snow-melt takes place; minor evapotranspiration

    Args:
        mask (integer): mask of 0 and 1 where 1 indicates pixels in this ET regime, shape (im_height,im_width)
        kc_list (list): crop coefficients for water requirements
        eto (float): daily value of reference evapotranspiration
        sb_old (float): snow bucket value from the previous day
        pr (float): daily value of precipitation
        wb_old (float): water bucket value from the previous day
        Sa (int): total available soil water holding capacity
        D (int): rooting depth
        p (int): the share of exess water, below which soil moisture starts to become difficult to extract
        Fsnm (float): snow melt parameter        
        tx (float): daily value of maximum temperature
        tmelt (float): the maximum temperature threshold, underwhich precip. falls as snow

    Returns:
        etm (float): daily value of the 'Maximum Evapotranspiration' (mm), shape (im_height,im_width)
        Eta (float): daily value of the 'Actual Evapotranspiration' (mm), shape (im_height,im_width)
        wb (float): daily value of the 'Soil Water Balance', shape (im_height,im_width)
        wx (float): daily value of the 'Maximum water available to plants', shape (im_height,im_width)
        sb (float): daily value of the 'Snow balance' (mm), shape (im_height,im_width)
        kc (float): daily value of the 'crop coefficients for water requirements', shape (im_height,im_width)
        """
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

    # snm = np.where(Fsnm*(tx - tmelt) > sb_old, sb_old, Fsnm*(tx - tmelt))   
    snm=np.zeros(sb_old.shape,dtype='float32')
    snm=np.where((sb_old>0.) & (Fsnm*(tx-tmelt)>sb_old),sb_old, snm) 
    snm=np.where((sb_old>0.) & (Fsnm*(tx-tmelt)<sb_old),Fsnm*(tx-tmelt),snm)     
    del tx
    sbx=sb_old-snm 
    del sb_old
    Salim = Sa*D

    sb=np.where(sbx>=etm,sbx-etm,0)

    wb, wx, Eta = eta(mask,wb_old+snm, etm, Sa, D, p, pr)
    del mask, p, wx

    Eta=np.where(sbx>=etm,etm,Eta)
    wb=np.where(sbx>=etm,wb_old+snm+pr-etm,wb)
    del pr#,snm 
    wb=np.where((sbx>=etm) & (wb>Salim),Salim,wb)
    # wx=np.where((sbx>=etm) & (wb>Salim),wb-Salim,wx)
    # wx=np.where((sbx>=etm) & (wb<=Salim),0,wx)

    wb=np.where(wb<0,0,wb)

    # print('snmelt',mask.dtype,eto.dtype,sb_old.dtype,pr.dtype,wb_old.dtype,p.dtype,tx.dtype,kc.dtype,snm.dtype,sbx.dtype,etm.dtype,Eta.dtype,wb.dtype,sb.dtype)
    return [etm, Eta, wb, sb]#,snm]


def EtaCalc_cold(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):
    """compute actual evapotranspiration (ETa) for ET regime "class 3"
       where there are Biological activities before start of growing period

    Args:
        mask (integer): mask of 0 and 1 where 1 indicates pixels in this ET regime, shape (im_height,im_width)
        kc_list (list): crop coefficients for water requirements
        eto (float): daily value of reference evapotranspiration
        sb_old (float): snow bucket value from the previous day
        pr (float): daily value of precipitation
        wb_old (float): water bucket value from the previous day
        Sa (int): total available soil water holding capacity
        D (int): rooting depth
        p (int): the share of exess water, below which soil moisture starts to become difficult to extract
        Fsnm (float): snow melt parameter        
        tx (float): daily value of maximum temperature
        tmelt (float): the maximum temperature threshold, underwhich precip. falls as snow

    Returns:
        etm (float): daily value of the 'Maximum Evapotranspiration' (mm), shape (im_height,im_width)
        Eta (float): daily value of the 'Actual Evapotranspiration' (mm), shape (im_height,im_width)
        wb (float): daily value of the 'Soil Water Balance', shape (im_height,im_width)
        wx (float): daily value of the 'Maximum water available to plants', shape (im_height,im_width)
        sb (float): daily value of the 'Snow balance' (mm), shape (im_height,im_width)
        kc (float): daily value of the 'crop coefficients for water requirements', shape (im_height,im_width)
        """
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

    # snm = np.where((Fsnm*(tx - tmelt) > sb_old), sb_old,Fsnm*(tx - tmelt)) 
    snm=np.zeros(sb_old.shape,dtype='float32')
    snm=np.where((sb_old>0.) & (Fsnm*(tx-tmelt)>sb_old),sb_old, snm) 
    snm=np.where((sb_old>0.) & (Fsnm*(tx-tmelt)<sb_old),Fsnm*(tx-tmelt),snm)   
    snm = np.where(sb_old<=0, 0, snm)   
    sb=sb_old-snm 
    del tx,sb_old

    wb, wx, Eta = eta(mask,wb_old+snm, etm, Sa, D, p, pr)
    del mask, wb_old, p, pr, wx#, snm

    Eta=np.where(Eta>etm,etm,Eta)

    # print('cold',mask.dtype,eto.dtype,sb_old.dtype,pr.dtype,wb_old.dtype,p.dtype,tx.dtype,kc.dtype,snm.dtype,etm.dtype,Eta.dtype,wb.dtype,sb.dtype)
    return [etm, Eta, wb, sb]#,snm]


def EtaCalc_warm(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt,istart,iend,idoy):
    """compute actual evapotranspiration (ETa) for ET regime "class 4" during the growing season

    Args:
        mask (integer): mask of 0 and 1 where 1 indicates pixels in this ET regime, shape (im_height,im_width)
        kc_list (list): crop coefficients for water requirements
        eto (float): daily value of reference evapotranspiration
        sb_old (float): snow bucket value from the previous day
        pr (float): daily value of precipitation
        wb_old (float): water bucket value from the previous day
        Sa (int): total available soil water holding capacity
        D (int): rooting depth
        p (int): the share of exess water, below which soil moisture starts to become difficult to extract
        Fsnm (float): snow melt parameter        
        tx (float): daily value of maximum temperature
        tmelt (float): maximum temperature threshold underwhich precip falls as snow
        istart0 (int): starting date of the growing period
        istart1 (int): ending date of the growing period
        idoy (int): index of the time loop
    Returns:
        etm (float): daily value of the 'Maximum Evapotranspiration' (mm), shape (im_height,im_width)
        Eta (float): daily value of the 'Actual Evapotranspiration' (mm), shape (im_height,im_width)
        wb (float): daily value of the 'Soil Water Balance', shape (im_height,im_width)
        wx (float): daily value of the 'Maximum water available to plants', shape (im_height,im_width)
        sb (float): daily value of the 'Snow balance' (mm), shape (im_height,im_width)
        kc (float): daily value of the 'crop coefficients for water requirements', shape (im_height,im_width)
        """    
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
    # snm=np.where(Fsnm*(tx-tmelt)>sb_old, sb_old, Fsnm*(tx-tmelt))
    # snm=np.where(sb_old<=0,0,snm)
    del tx
    sb=sb_old-snm
    del sb_old

    wb, wx, Eta = eta(mask,wb_old+snm, etm, Sa, D, p, pr)
    del mask,wb_old,p,pr,wx#,snm
    Eta=np.where(Eta>etm,etm,Eta)

    # print('warm',mask.dtype,eto.dtype,sb_old.dtype,pr.dtype,wb_old.dtype,p.dtype,tx.dtype,kc.dtype,snm.dtype,etm.dtype,Eta.dtype,wb.dtype,sb.dtype)
    return [etm, Eta, wb, sb]#,snm]


def EtaCalc_warmest(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):
    """compute actual evapotranspiration (ETa) for ET regime "class 5" the warmest period

    Args:
        mask (integer): mask of 0 and 1 where 1 indicates pixels in this ET regime, shape (im_height,im_width)
        kc_list (list): crop coefficients for water requirements
        eto (float): daily value of reference evapotranspiration
        sb_old (float): snow bucket value from the previous day
        pr (float): daily value of precipitation
        wb_old (float): water bucket value from the previous day
        Sa (int): total available soil water holding capacity
        D (int): rooting depth
        p (int): the share of exess water, below which soil moisture starts to become difficult to extract
        Fsnm (float): snow melt parameter        
        tx (float): daily value of maximum temperature
        tmelt (float): the maximum temperature threshold, underwhich precip. falls as snow

    Returns:
        etm (float): daily value of the 'Maximum Evapotranspiration' (mm), shape (im_height,im_width)
        Eta (float): daily value of the 'Actual Evapotranspiration' (mm), shape (im_height,im_width)
        wb (float): daily value of the 'Soil Water Balance', shape (im_height,im_width)
        wx (float): daily value of the 'Maximum water available to plants', shape (im_height,im_width)
        sb (float): daily value of the 'Snow balance' (mm), shape (im_height,im_width)
        kc (float): daily value of the 'crop coefficients for water requirements', shape (im_height,im_width)
        """
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
    # snm=np.where(Fsnm*(tx-tmelt)>sb_old, sb_old, Fsnm*(tx-tmelt))
    # snm=np.where(sb_old<=0,0,snm)
    del tx
    sb=sb_old-snm #????
    del sb_old#,snm
    
    wb, wx, Eta = eta(mask,wb_old, etm, Sa, D, p, pr)
    del mask,wb_old,p,pr,wx

    Eta=np.where(Eta>etm,etm,Eta)

    # print('warmest',mask.dtype,eto.dtype,sb_old.dtype,pr.dtype,wb_old.dtype,p.dtype,tx.dtype,kc.dtype,snm.dtype,etm.dtype,Eta.dtype,wb.dtype,sb.dtype)
    return [etm, Eta, wb, sb]#,snm]


def EtaCalc(im_mask,Tx,islgp,Pr,Txsnm,Fsnm,Eto,wb_old,sb_old,istart0,istart1,Sa,D,p,kc_list,lgpt5,eta_class,doy_start,doy_end,parallel):

    """vectorized calculation of actual evapotranspiration (ETa)
    
    Args:
        im_mask (integer): administrative mask of 0 and 1, if no mask was created by the user this comes in as all 1's, shape (im_height,im_width)
        Tx (float): daily value of maximum temperature
        Ta (float): daily value of average temperature
        Pr (float): daily value of precipitation
        Txsnm (float): maximum temperature threshold underwhich precip falls as snow
        Fsnm (float): snow melt parameter        
        Eto (float): daily value of reference evapotranspiration
        wb_old (float): water bucket value from the previous day
        sb_old (float): snow bucket value from the previous day
        idoy (int): index of the time loop
        istart0 (int): starting date of the growing period
        istart1 (int): ending date of the growing period
        Sa (int): total available soil water holding capacity
        D (int): rooting depth
        p (int): the share of exess water, below which soil moisture starts to become difficult to extract
        kc_list (list): crop coefficients for water requirements
        lgpt5 (float): number of days with mean daily tenmperature above 5 degC
    
    Returns:
        Etm_new (float): daily value of the 'Maximum Evapotranspiration' (mm), shape (im_height,im_width)
        Eta_new (float): daily value of the 'Actual Evapotranspiration' (mm), shape (im_height,im_width)
        Wb_new (float): daily value of the 'Soil Water Balance', shape (im_height,im_width)
        Wx_new (float): daily value of the 'Maximum water available to plants', shape (im_height,im_width)
        Sb_new (float): daily value of the 'Snow balance' (mm), shape (im_height,im_width)
        kc_new (float): daily value of the 'crop coefficients for water requirements', shape (im_height,im_width)        
    """  
    # we parallelize here by eta_class (the six different regimes for computing ET)
    # the computations for eta_class are each a delayed function
    # we call each function which saves the future computation as an object to a list of tasks
    # then we call compute on the list of tasks to execute them in parallel 
    if parallel:
        import dask

    ETM_list=[]  # list to hold 365 arrays (results for each day)
    ETA_list=[]  # list to hold 365 arrays (results for each day)
    # WB_list=[]   # list to hold 365 arrays (results for each day)
    # SB_list=[]   # list to hold 365 arrays (results for each day)
    # snm_list=[]
    # times={}
    # compute_times=[]
    # agg_times=[]

    # start0=timer()
    for idoy in range(doy_start-1, doy_end):

        # compute etm,eta,wb,sb for each eta regime in parallel each day
        if parallel:      
            # start=timer()
            results_list=[]  # list of lists: [[etm, eta, wb, sb],[etm, eta, wb, sb],...] in that order
        
            # delay data inputs so they're passed only once to the computations
            eta_class_d=dask.delayed(eta_class[:,:,idoy])
            Eto_d=dask.delayed(Eto[:,:,idoy])
            sb_old_d=dask.delayed(sb_old)
            Pr_d=dask.delayed(Pr[:,:,idoy])
            wb_old_d=dask.delayed(wb_old)
            p_d=dask.delayed(p[:,:,idoy])
            Tx_d=dask.delayed(Tx[:,:,idoy])

            # delayed call to each ETACalc_xxxxx
            # vars1, vars2... are lists of etm,eta,wb,sb arrays, in that order
            vars1=dask.delayed(EtaCalc_snow)(eta_class_d,1,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d)
            vars2=dask.delayed(EtaCalc_snowmelting)(eta_class_d,2,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d,Fsnm,Tx_d,Txsnm)                                                                       
            vars3=dask.delayed(EtaCalc_cold)(eta_class_d,3,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d,Fsnm,Tx_d,Txsnm)                                                               
            vars4=dask.delayed(EtaCalc_warm)(eta_class_d,4,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d,Fsnm,Tx_d,Txsnm,istart0,istart1,idoy)                                        
            vars5=dask.delayed(EtaCalc_warmest)(eta_class_d,5,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d,Fsnm,Tx_d,Txsnm)
            task_list=[vars1,vars2,vars3,vars4,vars5]

            # compute in parallel
            results_list=dask.compute(*task_list)
            # compute_times.append(timer()-start)

            # aggregate results (spatially for this idoy) from the 5 different ETACalc routines for each variable
            # start=timer()
            arr_shape=results_list[0][0].shape
            ETM_agg=np.empty(arr_shape,dtype='float32')
            ETA_agg=np.empty(arr_shape,dtype='float32')
            WB_agg=np.empty(arr_shape,dtype='float32')
            SB_agg=np.empty(arr_shape,dtype='float32')
            # snm_agg=np.empty(arr_shape,dtype='float32')
            # ETM_agg[:],ETA_agg[:],WB_agg[:],SB_agg[:],snm_agg[:]=np.float32(np.nan),np.float32(np.nan),np.float32(np.nan),np.float32(np.nan),np.float32(np.nan)
            ETM_agg[:],ETA_agg[:],WB_agg[:],SB_agg[:]=np.float32(np.nan),np.float32(np.nan),np.float32(np.nan),np.float32(np.nan)
            for icat,results in enumerate(results_list):
                ETM_agg=np.where(eta_class[:,:,idoy]==icat+1,results[0],ETM_agg)
                ETA_agg=np.where(eta_class[:,:,idoy]==icat+1,results[1],ETA_agg)
                WB_agg=np.where(eta_class[:,:,idoy]==icat+1,results[2],WB_agg)
                SB_agg=np.where(eta_class[:,:,idoy]==icat+1,results[3],SB_agg)
                # snm_agg=np.where(eta_class[:,:,idoy]==icat+1,results[4],snm_agg)
            # agg_times.append(timer()-start)            

        else:
            # start=timer()
            results_list=[] # list of lists: [[etm, eta, wb, sb],[etm, eta, wb, sb],...] in that order

            # call to each ETACalc_xxxxx    
            # vars1, vars2... are lists of etm,eta,wb,sb arrays, in that order
            vars1=EtaCalc_snow(eta_class[:,:,idoy],1,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy])
            vars2=EtaCalc_snowmelting(eta_class[:,:,idoy],2,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm)                                                                       
            vars3=EtaCalc_cold(eta_class[:,:,idoy],3,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm)                                                               
            vars4=EtaCalc_warm(eta_class[:,:,idoy],4,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm,istart0,istart1,idoy)                                        
            vars5=EtaCalc_warmest(eta_class[:,:,idoy],5,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm)
            results_list.append([vars1,vars2,vars3,vars4,vars5])            
            # compute_times.append(timer()-start)

            # aggregate results from the 5 different ETACalc routines for each variable
            # start=timer()
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
            # agg_times.append(timer()-start)
        
        # collect the results for each day in a list
        ETM_list.append(ETM_agg)
        ETA_list.append(ETA_agg)
        # WB_list.append(WB_agg)
        # SB_list.append(SB_agg)
        # snm_list.append(snm_agg)

        # update wb and sb values
        wb_old=WB_agg.copy()
        sb_old=SB_agg.copy()

        if parallel:
            del eta_class_d, Eto_d, sb_old_d, Pr_d, wb_old_d, p_d, Tx_d
        
        del vars1, vars2, vars3, vars4, vars5, task_list, results_list
        del ETM_agg, ETA_agg, WB_agg, SB_agg#, snm_agg

    # task_time0=timer()-start0

    # times['chunk_size']=Tx.shape
    # times['complete day loop']=task_time0   
    # times['avg daily compute'] = np.array(compute_times).mean()
    # times['avg daily agg'] = np.array(agg_times).mean()
    
    # clean up to release RAM
    del im_mask,Txsnm,Fsnm,Sa,D,kc_list
    del lgpt5,eta_class,istart0,istart1
    del Eto, sb_old, Pr, wb_old, p, Tx
    # create single array (all days together) for ETM, ETA
    # start=timer()
    ETM=np.stack(ETM_list,axis=-1,dtype='float32') # stack on last dim (time)
    ETA=np.stack(ETA_list,axis=-1,dtype='float32') # stack on last dim (time)
    # WB=np.stack(WB_list,axis=-1,dtype='float32') # stack on last dim (time)
    # SB=np.stack(SB_list,axis=-1,dtype='float32') # stack on last dim (time)
    # SNM=np.stack(snm_list,axis=-1,dtype='float32') # stack on last dim (time)
    # print('ETM,ETA info',ETM.shape,ETM.dtype,ETA.shape,ETA.dtype)
    # task_time=timer()-start
    # times['stack']=task_time
    # print('ETM,ETA1',ETM.dtype,ETA.dtype)     
    del ETM_list,ETA_list

    # eliminate negatives
    # start=timer()
    ETA=np.where(ETA<0,0,ETA)
    # task_time=timer()-start
    # times['ETA where']=task_time  
    # print('ETA2',ETA.dtype)

    # extend ETM,ETA so that the 10day rolling mean in
    # the next step doesn't reduce the time dimension
    # start=timer()
    with np.errstate(invalid='ignore'):  
        ETMx = np.append(ETM, ETM[:,:,0:12],axis=2)  
        ETAx = np.append(ETA, ETA[:,:,0:12],axis=2)  
    # print('ETMx,ETAx info',ETMx.shape,ETMx.dtype,ETAx.shape,ETAx.dtype)
    # task_time=timer()-start
    # times['append']=task_time  
    # print('ETMx,ETAx',ETMx.dtype,ETAx.dtype)
    del ETM, ETA
    # print(ETAx.shape,ETMx.shape)

    # compute forward-looking 10day rolling mean
    # start=timer()
    xx = val10day(ETAx)  
    yy = val10day(ETMx)  
    # task_time=timer()-start
    # times['xx,yy']=task_time 
    # print('xx,yy',xx.dtype,yy.dtype)
    del ETMx,ETAx

    # compute growing season length
    # start=timer()
    with np.errstate(divide='ignore', invalid='ignore'):  
        lgp_whole = xx[:,:,:doy_end]/yy[:,:,:doy_end]  
    # task_time=timer()-start
    # times['lgp_whole']=task_time 
    del xx,yy
    # print('lgp_whole',lgp_whole.dtype)

    # start=timer()
    lgp_tot=np.where((islgp==1)&(lgp_whole>=0.4),1,0).sum(axis=2)  
    # task_time=timer()-start
    # times['lgp_tot']=task_time   
    del lgp_whole   
    # print('lgp_tot',lgp_tot.dtype)
 
    # print(times) 
    return lgp_tot  
    # return ETM
    # return yy[:,:,:doy_end]
    # return islgp

