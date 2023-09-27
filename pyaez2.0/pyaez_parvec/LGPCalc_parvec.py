
"""
PyAEZ: LGPCalc.py calculates the length of growing period (LGP)
2022: K.Boonma ref. GAEZv4
"""

import numpy as np
import UtilitiesCalc_parvec as UtilitiesCalc

def rainPeak(totalPrec_monthly,meanT_daily,lgpt5):
    """Scan the monthly precipitation for the month with peak rainfall over 3 months

    Args:
        totalPrec_monthly (float): value of monthly precipitation for a pixel
        meanT_daily (float): daily mean temperature for a pixel
        lgpt5_point (float): thermal length of growing period (T>5C) for a pixel

    Returns:
        int: month of the peak rainfall
        1D NumPy: the smoothened daily mean temperature curve
        int: the starting date of the growing period
        int: the ending date of the growing period
    """    
    #============================================
    # Find peak rain (over 3 months)
    # rainmax = np.zeros((np.shape(totalPrec_monthly)))
    # rainmax[:,:,0] = totalPrec_monthly[:,:,11]+totalPrec_monthly[:,:,0]+totalPrec_monthly[:,:,1]

    # for i in range(1,11):
    #     rainmax[:,:,i] = totalPrec_monthly[:,:,i-1]+totalPrec_monthly[:,:,i]+totalPrec_monthly[:,:,i+1]
    # rainmax[:,:,11] = totalPrec_monthly[:,:,10]+totalPrec_monthly[:,:,11]+totalPrec_monthly[:,:,0]    

    # mon_rainmax=rainmax.argmax(axis=2)
    #============================================

    # get index of first occurrence in time where true at each grid cell
    day1=np.argmax(meanT_daily>=5.0,axis=2) 
    # argmax returns 0 where there is no first occurrence (no growing season) so need to fix
    day1=np.where(lgpt5==0,np.nan,day1)

    istart0=np.where((lgpt5<365),day1,0.) # replaces if block
    dat1=np.where(istart0>365,istart0-365,istart0)  # replaces setdat function
    istart1=np.where((lgpt5<365),dat1+lgpt5-1,lgpt5-1) # replaces if block

    return meanT_daily, istart0,istart1
#============================================
def isfromt0(meanT_daily_new,doy):
    """Check if the Julian day is coming from the temperature
       upward or downward trend

    Args:
        meanT_daily_new (1D NumPy): 1-year time-series of daily mean temperature
        doy (int): Julian day

    Returns:
        _type_: _description_
    """    
    if meanT_daily_new[doy]-meanT_daily_new[doy-1] > 0.:
        fromt0 = 1.
    else:
        fromt0 = 0.

    return fromt0

#============================================

def eta(mask,wb_old,etm,Sa,D,p,rain):
    """Calculate actual evapotranspiration (ETa)

    Args:
        wb_old (float): daily water balance left from the previous day
        etm (float): maximul evapotranspiration
        Sa (float): Available soil moisture holding capacity [mm/m]
        D (float): rooting depth [m]
        p (float): soil moisture depletion fraction (0-1)
        rain (float): amount of rainfall

    Returns:
        float: a value for daily water balance
        float: a value for total available soil moisture
        float: the calculated actual evapotranspiration
    """    
    s=wb_old+rain
    wx=np.where(mask==1,np.zeros(mask.shape),np.nan)
    Salim = max(Sa*D,1.)
    wr = np.where(100*(1.-p)>Salim,Salim,100*(1.-p))    

    eta_local=np.where(rain>=etm,etm,np.nan)
    eta_local=np.where((s-wr>=etm) & ~np.isfinite(eta_local),etm,eta_local)
    rho=wb_old/wr
    eta_local=np.where((rain+rho*etm >etm) & (mask==1) & ~np.isfinite(eta_local),etm,eta_local)
    eta_local=np.where((rain+rho*etm <etm) & (mask==1) & ~np.isfinite(eta_local),(rain+rho*etm <etm),eta_local)

    wb=s-eta_local
    wb=np.where(wb>=Salim,Salim,wb)
    wx=np.where(wb>=Salim,wb-Salim,wx)
    wx=np.where(wb<Salim,0,wx)
    wb=np.where(wb<0,0,wb)
    return wb, wx, eta_local

def psh(ng,et0):
    """Calculate soil moisture depletion fraction (0-1)

    Args:
        ng (float): crop group
        et0 (float): potential evapotranspiration [mm/day]

    Returns:
        float: soil moisture depletion fraction
    """    
    #ng = crop group
    # eto = potential evapotranspiration [mm/day]
    psh0=np.where(ng==0,0.5,0.3+(ng-1)*.05)  # replaces if block
    psh=psh0 +0.4*(5. - et0)
    psh=np.where(psh<0.1,0.1,psh)  # replaces if
    psh=np.where(psh>0.8,0.8,psh)  # replaces elif
    
    return psh

def val10day(Et):
    """Calculate 10-day moving average 

    Args:
        
    """
    # Program to calculate moving average along 1 dim of multidimensional array
    # use numpy convolve to get a forward-looking rolling mean on 1 dimension of a 3D array
    window_size = 10
    # first reshape the input data to 2d (space, time)
    Et2d=np.reshape(Et,(Et.shape[0]*Et.shape[1],Et.shape[2]))
    # use apply_along_axis to apply a rolling mean with length=window_size
    rollmean2d=np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size), mode='valid')/window_size, axis=1, arr=Et2d) 
    # reshape the result back to 3 dimensions
    rollmean3d=np.reshape(rollmean2d,(Et.shape[0],Et.shape[1],rollmean2d.shape[1]))

    return rollmean3d

def Eta_class(mask,lgpt5,ta,tx,tmelt):
    # initialize array to nan
    eta_class=np.empty(mask.shape,dtype='float32')
    eta_class[:]=np.nan

    # assign categorical value to snow areas
    # eta_class=np.where((ta<=0) & (tx<=tmelt) & ~np.isfinite(eta_class),1,eta_class)
    eta_class=np.where((ta<=0) & (tx<=tmelt),1,eta_class)

    # assign categorical value to snow melting areas
    eta_class=np.where((ta<=0) & (tx>=0) & ~np.isfinite(eta_class),2,eta_class)

    # assign categorical value to cold areas
    eta_class=np.where((ta>0) & (ta<5) & ~np.isfinite(eta_class),3,eta_class)

    # assign categorical value to warm areas
    eta_class=np.where((lgpt5<365) & (ta>=5) & ~np.isfinite(eta_class),4,eta_class)

    # assign categorical value to warmest areas
    eta_class=np.where((mask==1) & ~np.isfinite(eta_class),5,eta_class)

    return eta_class

def EtaCalc_snow(mask,kc_list,eto,sb_old,pr,wb_old,Sa,D,p):
    kc=np.where(mask==1,kc_list[0],np.nan)   
    etm = kc * eto
    sbx = sb_old + pr     

    # call the eta subroutine
    wb, wx, Eta = eta(mask,wb_old-pr, etm, Sa, D, p, pr)

    Salim=Sa*D
    sb=np.where(sbx>=etm,sbx-etm,sbx)
    sb=np.where(sbx<etm,0,sbx)  
    Eta=np.where(sbx>=etm,etm,Eta)

    wb=np.where(sbx>=etm,wb_old-etm,wb)
    wb=np.where((sbx>=etm) & (wb>Salim),Salim,wb)
    wx=np.where((sbx>=etm) & (wb>Salim),wb-Salim,wx)
    wx=np.where((sbx>=etm) & (wb<=Salim),0,wx) 
    return etm, Eta, wb, wx, sb, kc

def EtaCalc_snowmelting(mask,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):

    kc=np.where(mask==1,kc_list[1],np.nan)   
    etm = kc * eto

    snm = np.where(Fsnm*(tx - tmelt) > sb_old, sb_old, Fsnm*(tx - tmelt))   
    sbx=sb_old-snm 
    Salim = Sa*D

    sb=np.where(sbx>=etm,sbx-etm,sbx)
    sb=np.where(sbx<etm,0,sbx)

    wb, wx, Eta = eta(mask,wb_old+snm, etm, Sa, D, p, pr)

    Eta=np.where(sbx>=etm,etm,Eta)
    wb=np.where(sbx>=etm,wb_old+snm+pr-etm,wb)
    wb=np.where((sbx>=etm) & (wb>Salim),Salim,wb)
    wx=np.where((sbx>=etm) & (wb>Salim),wb-Salim,wx)
    wx=np.where((sbx>=etm) & (wb<=Salim),0,wx)

    wb=np.where(wb<0,0,wb)
   
    return etm, Eta, wb, wx, sb, kc

def EtaCalc_cold(mask,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):

    kc=np.where(mask==1,kc_list[2],np.nan)   
    etm = kc * eto

    snm = np.where((Fsnm*(tx - tmelt) > sb_old), sb_old, Fsnm*(tx - tmelt))   
    snm = np.where(sb_old<=0, 0, snm)   

    sb=sb_old-snm 

    wb, wx, Eta = eta(mask,wb_old+snm, etm, Sa, D, p, pr)

    Eta=np.where(Eta>etm,etm,Eta)

    return etm, Eta, wb, wx, sb, kc

def EtaCalc_warm(mask,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt,istart,iend,idoy):
    
    kc3=np.where(mask==1,kc_list[3],np.nan) 
    kc4=np.where(mask==1,kc_list[4],np.nan) 
    xx=np.where((mask==1) & (idoy-istart>=0) & (iend-idoy>=0) & ((idoy-istart)/30.>1.), 1., (idoy-istart)/30.)
    kc=np.where((idoy-istart>=0), kc3*(1.-xx)+(kc4*xx), kc3)

    etm = kc * eto

    snm=np.where((sb_old>0) & (Fsnm*(tx-tmelt)>sb_old), sb_old, Fsnm*(tx-tmelt))
    snm=np.where(sb_old<=0,0,snm)

    sb=sb_old-snm

    wb, wx, Eta = eta(mask,wb_old+snm, etm, Sa, D, p, pr)

    Eta=np.where(Eta>etm,etm,Eta)

    return etm, Eta, wb, wx, sb, kc

def EtaCalc_warmest(mask,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):

    kc=np.where(mask==1,kc_list[4],np.nan)   
    etm = kc * eto

    snm=np.where((sb_old>0) & (Fsnm*(tx-tmelt)>sb_old), sb_old, Fsnm*(tx-tmelt))
    snm=np.where(sb_old<=0,0,snm)

    sb=sb_old-snm
    
    wb, wx, Eta = eta(mask,wb_old, etm, Sa, D, p, pr)

    Eta=np.where(Eta>etm,etm,Eta)
    
    return etm, Eta, wb, wx, sb, kc 


def EtaCalc(im_mask,Tx,Ta,Pr,Txsnm,Fsnm,Eto,wb_old,sb_old,idoy,istart0,istart1,Sa,D,p,kc_list,lgpt5):

    """Calculate actual evapotranspiration (ETa)
    """  
    # incoming variables are either constants, the kc_list, an x,y array of time-invariant quantities
    # or an x,y, array of a single time step for time variant quantities
    # outputs x,y, arrays at a single time step for Etm_new,Eta_new,Wb_new,Wx_new,Sb_new,kc_new
    eta_class=Eta_class(im_mask,lgpt5,Ta,Tx,Txsnm)

    # compute for snow class
    mask=np.where(eta_class==1,1,np.nan)
    Etm_new,Eta_new,Wb_new,Wx_new,Sb_new,kc_new=EtaCalc_snow(mask,
                                                    kc_list,
                                                    np.where(mask==1,Eto,np.nan),
                                                    np.where(mask==1,sb_old,np.nan),
                                                    np.where(mask==1,Pr,np.nan),
                                                    np.where(mask==1,wb_old,np.nan),
                                                    Sa,
                                                    D,
                                                    np.where(mask==1,p,np.nan))

    # compute for snow melting class
    mask=np.where(eta_class==2,1,np.nan)
    etm,eta_var,wb,wx,sb,kc=EtaCalc_snowmelting(mask,
                                            kc_list,
                                            np.where(mask==1,Eto,np.nan),
                                            np.where(mask==1,sb_old,np.nan),
                                            np.where(mask==1,Pr,np.nan),
                                            np.where(mask==1,wb_old,np.nan),
                                            Sa,
                                            D,
                                            np.where(mask==1,p,np.nan),
                                            Fsnm,
                                            np.where(mask==1,Tx,np.nan),
                                            Txsnm)                                            
    Etm_new=np.where(mask==1,etm,Etm_new)
    Eta_new=np.where(mask==1,eta_var,Eta_new)                                                                                        
    Wb_new=np.where(mask==1,wb,Wb_new)                                                                                        
    Wx_new=np.where(mask==1,wx,Wx_new)                                                                                        
    Sb_new=np.where(mask==1,sb,Sb_new)                                                                                        
    kc_new=np.where(mask==1,kc,kc_new) 

    # compute for cold class
    mask=np.where(eta_class==3,1,np.nan)
    etm,eta_var,wb,wx,sb,kc=EtaCalc_cold(mask,
                                        kc_list,
                                        np.where(mask==1,Eto,np.nan),
                                        np.where(mask==1,sb_old,np.nan),
                                        np.where(mask==1,Pr,np.nan),
                                        np.where(mask==1,wb_old,np.nan),
                                        Sa,
                                        D,
                                        np.where(mask==1,p,np.nan),
                                        Fsnm,
                                        np.where(mask==1,Tx,np.nan),
                                        Txsnm)                                            
    Etm_new=np.where(mask==1,etm,Etm_new)
    Eta_new=np.where(mask==1,eta_var,Eta_new)                                                                                        
    Wb_new=np.where(mask==1,wb,Wb_new)                                                                                        
    Wx_new=np.where(mask==1,wx,Wx_new)                                                                                        
    Sb_new=np.where(mask==1,sb,Sb_new)                                                                                        
    kc_new=np.where(mask==1,kc,kc_new)

    # compute for warm class
    mask=np.where(eta_class==4,1,np.nan)
    etm,eta_var,wb,wx,sb,kc=EtaCalc_warm(mask,
                                        kc_list,
                                        np.where(mask==1,Eto,np.nan),
                                        np.where(mask==1,sb_old,np.nan),
                                        np.where(mask==1,Pr,np.nan),
                                        np.where(mask==1,wb_old,np.nan),
                                        Sa,
                                        D,
                                        np.where(mask==1,p,np.nan),
                                        Fsnm,
                                        np.where(mask==1,Tx,np.nan),
                                        Txsnm,
                                        np.where(mask==1,istart0,np.nan),
                                        np.where(mask==1,istart1,np.nan),
                                        idoy)                                          
    Etm_new=np.where(mask==1,etm,Etm_new)
    Eta_new=np.where(mask==1,eta_var,Eta_new)                                                                                        
    Wb_new=np.where(mask==1,wb,Wb_new)                                                                                        
    Wx_new=np.where(mask==1,wx,Wx_new)                                                                                        
    Sb_new=np.where(mask==1,sb,Sb_new)                                                                                        
    kc_new=np.where(mask==1,kc,kc_new)

    # compute for warmest class
    mask=np.where(eta_class==5,1,np.nan)
    etm,eta_var,wb,wx,sb,kc=EtaCalc_warmest(mask,
                                            kc_list,
                                            np.where(mask==1,Eto,np.nan),
                                            np.where(mask==1,sb_old,np.nan),
                                            np.where(mask==1,Pr,np.nan),
                                            np.where(mask==1,wb_old,np.nan),
                                            Sa,
                                            D,
                                            np.where(mask==1,p,np.nan),
                                            Fsnm,
                                            np.where(mask==1,Tx,np.nan),
                                            Txsnm)                                   
    Etm_new=np.where(mask==1,etm,Etm_new)
    Eta_new=np.where(mask==1,eta_var,Eta_new)                                                                                        
    Wb_new=np.where(mask==1,wb,Wb_new)                                                                                        
    Wx_new=np.where(mask==1,wx,Wx_new)                                                                                        
    Sb_new=np.where(mask==1,sb,Sb_new)                                                                                        
    kc_new=np.where(mask==1,kc,kc_new)    

    return Etm_new,Eta_new,Wb_new,Wx_new,Sb_new,kc_new 


def getLGP_chunk(chunkid,im_mask,Ta365,Pcp365,Tx365,Eto365,lgpt5,Sa=100.,D=1.):
    #============================
    kc_list = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
    #============================
    Txsnm = 0.  # Txsnm - snow melt temperature threshold
    Fsnm = 5.5  # Fsnm - snow melting coefficient
    Sb_old = np.zeros(im_mask.shape)
    Wb_old = np.zeros(im_mask.shape)
    #============================
    Etm365=np.empty(Eto365.shape)
    Eta365=np.empty(Eto365.shape)
    Wb365=np.empty(Eto365.shape)
    Wx365=np.empty(Eto365.shape)
    Sb365=np.empty(Eto365.shape)
    kc365=np.empty(Eto365.shape)
    Etm365[:],Eta365[:],Wb365[:],Wx365[:],Sb365[:],kc365[:]=np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    #============================
    obj_utilities = UtilitiesCalc.UtilitiesCalc()
    totalPrec_monthly = obj_utilities.averageDailyToMonthly(Pcp365)
    meanT_daily_new, istart0, istart1 = rainPeak(totalPrec_monthly, Ta365, lgpt5)
    p = psh(np.zeros(Eto365.shape), Eto365)            

    for doy in range(0, 365):
        Eta_new, Etm_new, Wb_new, Wx_new, Sb_new, kc_new = EtaCalc(
                                im_mask,
                                np.float64(Tx365[:,:,doy]), 
                                np.float64(Ta365[:,:,doy]),
                                np.float64(Pcp365[:,:,doy]), 
                                Txsnm, 
                                Fsnm, 
                                np.float64(Eto365[:,:,doy]),
                                Wb_old, 
                                Sb_old, 
                                doy, 
                                istart0, 
                                istart1,
                                Sa, 
                                D, 
                                p[:,:,doy], 
                                kc_list, 
                                lgpt5)

        Etm365[:,:,doy]=Etm_new
        Eta365[:,:,doy]=Eta_new
        Wb365[:,:,doy]=Wb_new
        Wx365[:,:,doy]=Wx_new
        Sb365[:,:,doy]=Sb_new
        kc365[:,:,doy]=kc_new

        Wb_old=Wb_new
        Sb_old=Sb_new

    Eta365=np.where(Eta365<0,0,Eta365)  

    Etm365X = np.append(Etm365, Etm365[:,:,0:30],axis=2)
    Eta365X = np.append(Eta365, Eta365[:,:,0:30],axis=2)

    # eliminate call to LGPCalc.islgpt
    islgp=np.where(Ta365>=5,1,0)
    
    xx = val10day(Eta365X)
    yy = val10day(Etm365X)

    with np.errstate(divide='ignore', invalid='ignore'):
        lgp_whole = xx[:,:,:365]/yy[:,:,:365]

    lgp_tot=np.where((islgp==1)&(lgp_whole>=0.4),1,0).sum(axis=2)
    lgp_tot=np.where(im_mask==1,lgp_tot,np.nan)    

    return {chunkid:[Etm365,Eta365,Wb365,Wx365,Sb365,kc365,lgp_tot]}    




