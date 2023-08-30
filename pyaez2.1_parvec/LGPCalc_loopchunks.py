"""
PyAEZ version 2.1.0 (June 2023)
PyAEZ: LGPCalc.py calculates the length of growing period (LGP)
2022/2023: Kittiphon Boonma 
"""

# from numba import jit
import numpy as np
# import dask
# import psutil
from time import time as timer
# import dask.array as da
# import sys


# @jit(nopython=True)
def rainPeak(meanT_daily, lgpt5):
# def rainPeak(totalPrec_monthly, meanT_daily, lgpt5_point):
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
    # # ============================================
    # days_f = np.arange(0, 365)
    # lgpt5_veg = days_f[meanT_daily >= 5]
    # # ============================================
    # if lgpt5_point < 365:
    #     istart0 = lgpt5_veg[0]
    #     istart1 = setdat(istart0) + lgpt5_point-1
    # else:
    #     istart0 = 0
    #     istart1 = lgpt5_point-1

    # vectorization  #KLG
    # get index of first occurrence in time where true at each grid cell  #KLG
    day1=np.argmax(meanT_daily>=5.0,axis=2)   #KLG
    # argmax returns 0 where there is no first occurrence (no growing season) so need to fix  #KLG
    day1=np.where(lgpt5==0,np.nan,day1)  #KLG
    istart0=np.where((lgpt5<365),day1,0).astype('float32') # replaces if block  #KLG
    dat1=np.where(istart0>365,istart0-365,istart0)  # replaces setdat function  #KLG
    istart1=np.where((lgpt5<365),dat1+lgpt5-1,lgpt5-1).astype('float32') # replaces if block  #KLG

    
    # day1=da.argmax(meanT_daily>=5.0,axis=2)   #KLG
    # day1=da.where(lgpt5==0,np.nan,day1)  #KLG
    # istart0=da.where((lgpt5<365),day1,0).astype('float32')
    # # istart0=da.where(mask,istart0,np.nan).astype('float32')
    # dat1=da.where(istart0>365,istart0-365,istart0)  # replaces setdat function  #KLG
    # istart1=da.where((lgpt5<365),dat1+lgpt5-1,lgpt5-1).astype('float32') # replaces if block  #KLG

    return istart0, istart1
# ============================================


# @jit(nopython=True)
# def isfromt0(meanT_daily_new, doy):
#     """Check if the Julian day is coming from the temperature
#        upward or downward trend

#     Args:
#         meanT_daily_new (1D NumPy): 1-year time-series of daily mean temperature
#         doy (int): Julian day

#     Returns:
#         _type_: _description_
#     """
#     if meanT_daily_new[doy]-meanT_daily_new[doy-1] > 0.:
#         fromt0 = 1.
#     else:
#         fromt0 = 0.

#     return fromt0

# ============================================


# @jit(nopython=True)
# def eta(wb_old, etm, Sa, D, p, rain):
def eta(mask,wb_old,etm,Sa,D,p,rain):  #KLG

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
    # wx = 0
    # wx=np.where(mask==1,np.zeros(mask.shape,dtype='float32'),np.float32(np.nan))  #KLG
    wx=np.where(mask==1,np.float32(0),np.float32(np.nan))  #KLG

    Salim = max(Sa*D, 1.)
    # wr=min(100*(1-p),Salim)
    # wr = np.where(100*(1.-p)>Salim,np.float32(Salim),np.float32(100*(1.-p)))    #KLG
    wr = np.where(100*(1.-p)>Salim,Salim,100*(1.-p))    #KLG
    # print('wx,wr',wx.dtype,wr.dtype)


    # if rain >= etm:
    #     eta = etm
    # elif s-wr >= etm:
    #     eta = etm
    # else:
    #     rho = wb_old/wr
    #     eta = min(rain + rho*etm, etm)
    eta_local=np.where(rain>=etm,np.float32(etm),np.float32(np.nan))  #KLG
    # print('eta_local1',eta_local.dtype)
    eta_local=np.where((s-wr>=etm)& ~np.isfinite(eta_local),np.float32(etm),np.float32(eta_local))  #KLG
    # eta_local=np.where((s-wr>=etm)& ~np.isfinite(eta_local),etm,eta_local)  #KLG    
    # print('eta_local2',eta_local.dtype)
    rho=wb_old/wr  #KLG
    # print('rho',rho.dtype)
    eta_local=np.where((rain+rho*etm >=etm) & (mask==1) & ~np.isfinite(eta_local),np.float32(etm),np.float32(eta_local))  #KLG
    # eta_local=np.where((rain+rho*etm >=etm) & (mask==1) & ~np.isfinite(eta_local),etm,eta_local)  #KLG
    # print('eta_local3',eta_local.dtype)
    eta_local=np.where((rain+rho*etm <etm) & (mask==1) & ~np.isfinite(eta_local),np.float32(rain+rho*etm),np.float32(eta_local))  #KLG
    # eta_local=np.where((rain+rho*etm <etm) & (mask==1) & ~np.isfinite(eta_local),rain+rho*etm,eta_local)  #KLG
    # print('eta_local4',eta_local.dtype)
    # wb=s-eta
    wb=s-eta_local  #KLG
    # print('wb1',wb.dtype)
    # if wb >= Salim:
    #     wx = wb-Salim
    #     wb = Salim
    # else:
    #     wx = 0
    # wb=np.where(wb>=Salim,np.float32(Salim),np.float32(wb))  #KLG
    # wx=np.where(wb>=Salim,np.float32(wb-Salim),np.float32(wx))  #KLG
    # wx=np.where(wb<Salim,np.float32(0),np.float32(wx))    #KLG      
    wb=np.where(wb>=Salim,Salim,wb)  #KLG
    # print('wb2',wb.dtype)
    wx=np.where(wb>=Salim,wb-Salim,wx)  #KLG
    # print('wx2',wx.dtype)
    wx=np.where(wb<Salim,0,wx)    #KLG        
    # print('wx3',wx.dtype)
    # if wb < 0:
    #     wb=0
    # wb=np.where(wb<0,np.float32(0),np.float32(wb))  #KLG
    wb=np.where(wb<0,0,wb)  #KLG
    # print('wb4',wb.dtype)



    # return wb, wx, eta
    # print(wb.dtype,wx.dtype,eta_local.dtype)
    return wb, wx, eta_local  #KLG


# @jit(nopython=True)
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

    # psh0=da.where(ng==0,0.5,0.3+(ng-1)*.05)
    # psh = psh0 + .04 * (5.-et0)
    # psh=da.where(psh<0.1,0.1,psh)  
    # psh=da.where(psh>0.8,0.8,psh)

    return psh.astype('float32')


# @jit(nopython=True)
def val10day(Et):
    """Calculate 10-day moving average 

    Args:
        
    """
    # Program to calculate moving average
    # arr = Et
    # window_size = 10
    # i = 0
    # Initialize an empty list to store moving averages
    # moving_averages = []

    # # Loop through the array to consider
    # # every window of size 3
    # while i < len(arr) - window_size + 1:
    #     # Store elements from i to i+window_size
    #     # in list to get the current window
    #     window = arr[i: i + window_size]
    #     # Calculate the average of current window
    #     window_average = round(sum(window) / window_size, 2)
    #     # Store the average of current
    #     # window in moving average list
    #     moving_averages.append(window_average)
    #     # Shift window to right by one position
    #     i += 1

    # # this is too slow
    # # use numpy convolve to get a forward-looking rolling mean on 1 dimension of a 3D array
    # window_size = 10
    # # first reshape the input data to 2d (space, time)
    # Et2d=np.reshape(Et,(Et.shape[0]*Et.shape[1],Et.shape[2])).astype('float32')
    # # use apply_along_axis to apply a rolling mean with length=window_size
    # rollmean2d=np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size), mode='valid')/window_size, axis=1, arr=Et2d) 
    # # reshape the result back to 3 dimensions
    # rollmean3d=np.reshape(rollmean2d,(Et.shape[0],Et.shape[1],rollmean2d.shape[1])).astype('float32')
    window_size = 10
    var = np.cumsum(Et, axis=-1, dtype='float32')
    var[:,:,window_size:] = var[:,:,window_size:] - var[:,:,:-window_size]
    rollmean3d = np.round((var[:,:,window_size-1:]/window_size),2)
    # print(rollmean3d.dtype)


    # return np.array(moving_averages)
    return rollmean3d


# @jit(nopython=True)
# def EtaCalc(Tx365, Ta365, Pcp365, Txsnm, Fsnm, Eto365, wb_old, sb_old, doy, istart0, istart1, Sa, D, p, kc_list, lgpt5_point):
#     """Calculate actual evapotranspiration (ETa)
#         This is a Numba routine, which means all the arguments are a single element -- not an array. 
#     Args:
#         Tx365 (float): a daily value of maximum temperature
#         Ta365 (float): a daily value of average temperature
#         Pcp365 (float): a daily value of precipitation
#         Txsnm (float): the maximum temperature threshold, underwhich precip. falls as snow
#         Fsnm (float): snow melt parameter
#         Eto365 (float): a daily value of reference evapotranspiration
#         wb_old (float): water bucket value from the previous day
#         sb_old (float): snow bucket value from the previous day
#         doy (int): day of year
#         istart0 (int): the starting date of the growing period
#         istart1 (int): the ending date of the growing period
#         Sa (int): total available soil water holding capacity
#         D (int): rooting depth
#         p (int): the share of exess water, below which soil moisture starts to become difficult to extract
#         kc_list (list): crop coefficients for water requirements
#         lgpt5_point (float): numbers of days with mean daily tenmperature above 5 degC

#     Returns:
#         Eta365 (float): a daily value of the 'Actual Evapotranspiration' (mm)
#         Etm365 (float): a daily value of the 'Maximum Evapotranspiration' (mm)
#         Wb365 (float): a daily value of the 'Soil Water Balance'
#         Wx365 (float): a daily value of the 'Maximum water available to plants'
#         Sb365 (float): a daily value of the 'Snow balance' (mm)
#         kc365 (float): a daily value of the 'crop coefficients for water requirements'
#     """

#     # Period with Tmax <= Txsnm (precipitaton falls as snow as is added to snow bucket)
#     if Tx365 <= Txsnm and Ta365 <= 0.:
#         etm = kc_list[0] * Eto365

#         Etm365 = etm

#         sbx = sb_old+Pcp365

#         wb, wx, Eta = eta(wb_old-Pcp365, etm, Sa, D, p, Pcp365)

#         Salim = Sa*D  

#         if sbx >= etm:
#             Sb365 = sbx-etm
#             Eta365 = etm

#             wb=wb_old-etm
#             if wb > Salim:
#                 wx = wb- Salim 
#                 wb = Salim
#             else:
#                 wx = 0
#         else:
#             Sb365 = 0.
#             Eta365 = Eta

#         if wb < 0:
#             wb = 0

#         Wb365 = wb
#         Wx365 = wx
#         kc365 = kc_list[0]

#     # Snow-melt takes place; minor evapotranspiration
#     elif Ta365 <= 0. and Tx365 >= 0.:
#         etm = kc_list[1] * Eto365
#         Etm365 = etm
#         ks = 0.1
#         # Snow-melt function
#         snm = min(Fsnm*(Tx365-Txsnm), sb_old)
#         sbx = sb_old - snm 
#         Salim = Sa*D
#         if sbx >= etm:
#             Sb365 = sbx-etm
#             Eta365 = etm
#             wb = wb_old+snm+Pcp365-etm

#             if wb > Salim:
#                 wx = wb - Salim
#                 wb = Salim
#             else:
#                 wx = 0
#         else:
#             Sb365 = 0.
#             wb, wx, Eta = eta(wb_old+snm, etm, Sa, D, p, Pcp365)
#             Eta365 = Eta

#         if wb < 0:
#             wb = 0
      
#         Wb365 = wb
#         Wx365 = wx
#         kc365 = kc_list[1]

#     elif Ta365 < 5. and Ta365 > 0.:
#         # Biological activities before start of growing period
#         etm = kc_list[2] * Eto365
#         Etm365 = etm

#         # In case there is still snow
#         if sb_old > 0.:
#             snm = min(Fsnm*(Tx365-Txsnm), sb_old)
#         else:
#             snm = 0.

#         sbx = sb_old-snm

#         wb, wx, Eta = eta(wb_old+snm, etm, Sa, D, p, Pcp365)

#         if Eta > Etm365:
#             Eta365 = etm
#         else:
#             Eta365=Eta

#         Wx365 = wx
#         Wb365 = wb
#         Sb365 = sbx
#         kc365 = kc_list[2]     

#     elif lgpt5_point < 365 and Ta365 >= 5.:
#         if doy >= istart0 and doy <= istart1:
#             # case 2 -- kc increases from 0.5 to 1.0 during first month of LGP
#             # case 3 -- kc = 1 until daily Ta falls below 5C
#             xx = min((doy-istart0)/30., 1.)
#             kc = kc_list[3]*(1.-xx)+(kc_list[4]*xx)
#         else:
#             # case 1 -- kc=0.5 for days until start of growing period
#             kc = kc_list[3]

#         etm = kc * Eto365
#         Etm365 = etm
#         # In case there is still snow
#         if sb_old > 0.:
#             snm = min(Fsnm*(Tx365-Txsnm), sb_old)
#         else:
#             snm = 0.

#         sbx = sb_old-snm

#         wb, wx, Eta = eta(wb_old+snm, etm, Sa, D, p, Pcp365)

#         if Eta > Etm365:
#             Eta365 = etm
#         else:
#             Eta365 = Eta

#         Wb365 = wb
#         Wx365 = wx
#         Sb365 = sbx
#         kc365 = kc
#     else:
#         etm = kc_list[4] * Eto365
#         Etm365 = etm
#         # In case there is still snow
#         if sb_old > 0.:
#             snm = min(Fsnm*(Tx365-Txsnm), sb_old)
#         else:
#             snm = 0.

#         wb, wx, Eta = eta(wb_old, etm, Sa, D, p, Pcp365)

#         if Eta > Etm365:
#             Eta365 = etm
#         else:
#             Eta365 = Eta
            
#         Wb365 = wb
#         Wx365 = wx
#         Sb365 = sbx
#         kc365 = kc_list[4]

#     return Eta365, Etm365, Wb365, Wx365, Sb365, kc365


# Here I'm dividing EtaCalc up where each part of the if block in the original EtaCalc is a separate function (I call this EtaClass)
# So EtaClass determines which section of the big if block applies to each pixel
# We send all pixels of each "class" using a mask to the appropriate EtaCalc_xxxx function and then combine the results together 
# I'm doing all this because we generally need to remove if blocks and replace with where statements for vectorization
# and also, eventually we can add parallelization where each part of the if block (EtaCalc_xxxx functions) are computed simultaneously 
# The flow is:
#    ClimateRegime.getLGP calls LGPCalc.EtaCalc, 
#    LGPCalc.EtaCalc calls Eta_class then EtaCalc_snow, EtaCalc_snowmelting, EtaCalc_cold, EtaCalc_warm, EtaCalc_warmest
#    LGPCalc.EtaCalc then assembles all results and returns the same 6 variables as the original EtaCalc function, but in 2D arrays (all pixels) instead of 1 pixel at a time

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
    # # initialize array to nan
    # # eta_class=np.empty(mask.shape,dtype='float32')
    # # eta_class[:]=np.nan

    # ta=ta.squeeze()
    # tx=tx.squeeze()

    # # assign categorical value to snow areas
    # eta_class=np.where((ta<=0) & (tx<=tmelt),np.float32(1),np.float32(np.nan))

    # # assign categorical value to snow melting areas
    # eta_class=np.where((ta<=0) & (tx>=0) & ~np.isfinite(eta_class),np.float32(2),np.float32(eta_class))

    # # assign categorical value to cold areas (pre-growing season)
    # eta_class=np.where((ta>0) & (ta<5) & ~np.isfinite(eta_class),np.float32(3),np.float32(eta_class))

    # # assign categorical value to warm areas (growing season)
    # eta_class=np.where((lgpt5<365) & (ta>=5) & ~np.isfinite(eta_class),np.float32(4),np.float32(eta_class))

    # # assign categorical value to warmest areas
    # eta_class=np.where((mask==1) & ~np.isfinite(eta_class),np.float32(5),np.float32(eta_class))
    # # print('in eta_class', psutil.virtual_memory().free/1E9)

    # eta_class=da.where((ta<=0) & (tx<=tmelt),np.float32(1),np.float32(np.nan))
    # eta_class=da.where((ta<=0) & (tx>=0) & ~np.isfinite(eta_class),np.float32(2),np.float32(eta_class))
    # eta_class=da.where((ta>0) & (ta<5) & ~np.isfinite(eta_class),np.float32(3),np.float32(eta_class))
    # eta_class=da.where((lgpt5<365) & (ta>=5) & ~np.isfinite(eta_class),np.float32(4),np.float32(eta_class))
    # eta_class=da.where((mask==1) & ~np.isfinite(eta_class),np.float32(5),np.float32(eta_class))    
    # ta=ta.squeeze()
    # tx=tx.squeeze()

    # eta_class=da.where((ta<=0)&(tx<=tmelt),1,np.nan)
    # eta_class=da.where((ta<=0)&(tx>=0)&(~np.isfinite(eta_class)),2,eta_class)
    # eta_class=da.where((ta>0)&(ta<5)&(~np.isfinite(eta_class)),3,eta_class)
    # eta_class=da.where((lgpt5<365)&(ta>=5)&(~np.isfinite(eta_class)),4,eta_class)
    # eta_class=da.where((mask==1)&(~np.isfinite(eta_class)),5,eta_class) 

    # eta_class=np.where((ta<=0)&(tx<=tmelt),1,0)
    # eta_class=np.where((ta<=0)&(tx>=0)&(eta_class==0),2,eta_class)
    # eta_class=np.where((ta>0)&(ta<5)&(eta_class==0),3,eta_class)
    # eta_class=np.where((lgpt5<365)&(ta>=5)&(eta_class==0),4,eta_class)
    # eta_class=np.where((mask==1)&(eta_class==0),5,eta_class) 

    # this method uses less memory
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

    # # return eta_class.astype('float32')
    # return eta_class.astype('int8')

# @dask.delayed
def EtaCalc_snow(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p):
# def EtaCalc_snow(mask,kc_list,eto,sb_old,pr,wb_old,Sa,D,p):
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
    # mask=np.where(classmap==classnum,np.float32(1),np.float32(np.nan))
    # eto=np.where(mask==1,np.float32(eto),np.float32(np.nan))
    # sb_old=np.where(mask==1,np.float32(sb_old),np.float32(np.nan))
    # pr=np.where(mask==1,np.float32(pr),np.float32(np.nan))
    # wb_old=np.where(mask==1,np.float32(wb_old),np.float32(np.nan))
    # p=np.where(mask==1,np.float32(p),np.float32(np.nan))
    mask=np.where(classmap==classnum,1,0)#.astype('int32')
    eto=np.where(mask==1,eto,np.float32(np.nan))
    sb_old=np.where(mask==1,sb_old,np.float32(np.nan))
    pr=np.where(mask==1,pr,np.float32(np.nan))
    wb_old=np.where(mask==1,wb_old,np.float32(np.nan))
    p=np.where(mask==1,p,np.float32(np.nan))


    # kc=np.where(mask==1,np.float32(kc_list[0]),np.float32(np.nan))  
    kc=np.zeros(mask.shape,dtype='float32')   
    kc=np.where(mask==1,kc_list[0],kc)   
    etm = kc * eto
    del eto, kc
    sbx = sb_old + pr 
    del sb_old    

    # call the eta subroutine
    wb, wx, Eta = eta(mask,wb_old-pr, etm, Sa, D, p, pr)
    del mask,pr,p,wx

    Salim=Sa*D
    # sb=np.where(sbx>=etm,np.float32(sbx-etm),np.float32(0))
    # Eta=np.where(sbx>=etm,np.float32(etm),np.float32(Eta))
    sb=np.where(sbx>=etm,sbx-etm,0)
    # sb=np.where(sbx>=etm,sbx-etm,sbx)
    # sb=np.where(sbx<etm,0,sb)
    Eta=np.where(sbx>=etm,etm,Eta)

    # wb=np.where(sbx>=etm,np.float32(wb_old-etm),np.float32(wb))
    # wb=np.where((sbx>=etm) & (wb>Salim),np.float32(Salim),np.float32(wb))
    wb=np.where(sbx>=etm,wb_old-etm,wb)
    wb=np.where((sbx>=etm) & (wb>Salim),Salim,wb)    
    # what are we using wx for
    # wx=np.where((sbx>=etm) & (wb>Salim),wb-Salim,wx)
    # wx=np.where((sbx>=etm) & (wb<=Salim),0,wx) 

    # return etm, Eta, wb, wx, sb, kc
    # return (etm, Eta, wb, sb)
    # print('snow',mask.dtype,eto.dtype,sb_old.dtype,pr.dtype,wb_old.dtype,p.dtype,kc.dtype,sbx.dtype,etm.dtype,Eta.dtype,wb.dtype,sb.dtype)
    return [etm, Eta, wb, sb]

# @dask.delayed
def EtaCalc_snowmelting(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):
# def EtaCalc_snowmelting(mask,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):
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
    # mask=np.where(classmap==classnum,np.float32(1),np.float32(np.nan))
    # eto=np.where(mask==1,np.float32(eto),np.float32(np.nan))
    # sb_old=np.where(mask==1,np.float32(sb_old),np.float32(np.nan))
    # pr=np.where(mask==1,np.float32(pr),np.float32(np.nan))
    # wb_old=np.where(mask==1,np.float32(wb_old),np.float32(np.nan))
    # p=np.where(mask==1,np.float32(p),np.float32(np.nan))
    # tx=np.where(mask==1,np.float32(tx),np.float32(np.nan))
    mask=np.where(classmap==classnum,1,0)#.astype('int32')
    eto=np.where(mask==1,eto,np.float32(np.nan))
    sb_old=np.where(mask==1,sb_old,np.float32(np.nan))
    pr=np.where(mask==1,pr,np.float32(np.nan))
    wb_old=np.where(mask==1,wb_old,np.float32(np.nan))
    p=np.where(mask==1,p,np.float32(np.nan))
    tx=np.where(mask==1,tx,np.float32(np.nan))

    # kc=np.where(mask==1,np.float32(kc_list[1]),np.float32(np.nan))
    kc=np.zeros(mask.shape,dtype='float32')   
    kc=np.where(mask==1,kc_list[1],kc)
    etm = kc * eto
    del eto, kc

    # snm = np.where(Fsnm*(tx - tmelt) > sb_old, np.float32(sb_old), np.float32(Fsnm*(tx - tmelt)))   
    snm = np.where(Fsnm*(tx - tmelt) > sb_old, sb_old, Fsnm*(tx - tmelt))   
    del tx
    sbx=sb_old-snm 
    del sb_old
    Salim = Sa*D

    # sb=np.where(sbx>=etm,np.float32(sbx-etm),np.float32(0))
    sb=np.where(sbx>=etm,sbx-etm,0)
    # sb=np.where(sbx>=etm,sbx-etm,sbx)
    # sb=np.where(sbx<etm,0,sb)    

    wb, wx, Eta = eta(mask,wb_old+snm, etm, Sa, D, p, pr)
    del mask, p, wx
    # print('wb,Eta',wb.dtype,Eta.dtype)
    # Eta=np.where(sbx>=etm,np.float32(etm),np.float32(Eta))
    # wb=np.where(sbx>=etm,np.float32(wb_old+snm+pr-etm),np.float32(wb))
    # wb=np.where((sbx>=etm) & (wb>Salim),np.float32(Salim),np.float32(wb))
    Eta=np.where(sbx>=etm,etm,Eta)
    wb=np.where(sbx>=etm,wb_old+snm+pr-etm,wb)
    del snm, pr
    wb=np.where((sbx>=etm) & (wb>Salim),Salim,wb)
    # wx=np.where((sbx>=etm) & (wb>Salim),wb-Salim,wx)
    # wx=np.where((sbx>=etm) & (wb<=Salim),0,wx)

    # wb=np.where(wb<0,np.float32(0),np.float32(wb))
    wb=np.where(wb<0,0,wb)
    # print(wb.dtype,Eta.dtype)
    # return etm, Eta, wb, wx, sb, kc
    # return (etm, Eta, wb, sb)

    # print('snmelt',mask.dtype,eto.dtype,sb_old.dtype,pr.dtype,wb_old.dtype,p.dtype,tx.dtype,kc.dtype,snm.dtype,sbx.dtype,etm.dtype,Eta.dtype,wb.dtype,sb.dtype)
    return [etm, Eta, wb, sb]

# @dask.delayed
def EtaCalc_cold(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):
# def EtaCalc_cold(mask,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):
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
    # mask=np.where(classmap==classnum,np.float32(1),np.float32(np.nan))
    # eto=np.where(mask==1,np.float32(eto),np.float32(np.nan))
    # sb_old=np.where(mask==1,np.float32(sb_old),np.float32(np.nan))
    # pr=np.where(mask==1,np.float32(pr),np.float32(np.nan))
    # wb_old=np.where(mask==1,np.float32(wb_old),np.float32(np.nan))
    # p=np.where(mask==1,np.float32(p),np.float32(np.nan))
    # tx=np.where(mask==1,np.float32(tx),np.float32(np.nan))
    mask=np.where(classmap==classnum,1,0)
    eto=np.where(mask==1,eto,np.float32(np.nan))
    sb_old=np.where(mask==1,sb_old,np.float32(np.nan))
    pr=np.where(mask==1,pr,np.float32(np.nan))
    wb_old=np.where(mask==1,wb_old,np.float32(np.nan))
    p=np.where(mask==1,p,np.float32(np.nan))
    tx=np.where(mask==1,tx,np.float32(np.nan))

    # kc=np.where(mask==1,np.float32(kc_list[2]),np.float32(np.nan))   
    kc=np.zeros(mask.shape,dtype='float32')   
    kc=np.where(mask==1,kc_list[2],kc)
    etm = kc * eto
    del eto, kc

    # snm = np.where((Fsnm*(tx - tmelt) > sb_old), np.float32(sb_old), np.float32(Fsnm*(tx - tmelt)))   
    # snm = np.where(sb_old<=0, np.float32(0), np.float32(snm))   
    snm = np.where((Fsnm*(tx - tmelt) > sb_old), sb_old,Fsnm*(tx - tmelt))   
    del tx
    snm = np.where(sb_old<=0, 0, snm)   
    sb=sb_old-snm 
    del sb_old

    wb, wx, Eta = eta(mask,wb_old+snm, etm, Sa, D, p, pr)
    del mask, wb_old, snm, p, pr, wx

    # Eta=np.where(Eta>etm,np.float32(etm),np.float32(Eta))
    Eta=np.where(Eta>etm,etm,Eta)

    # return etm, Eta, wb, wx, sb, kc
    # return (etm, Eta, wb, sb)
    # print('cold',mask.dtype,eto.dtype,sb_old.dtype,pr.dtype,wb_old.dtype,p.dtype,tx.dtype,kc.dtype,snm.dtype,etm.dtype,Eta.dtype,wb.dtype,sb.dtype)
    return [etm, Eta, wb, sb]

# @dask.delayed
def EtaCalc_warm(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt,istart,iend,idoy):
# def EtaCalc_warm(mask,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt,istart,iend,idoy):
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
    # mask=np.where(classmap==classnum,np.float32(1),np.float32(np.nan))
    # eto=np.where(mask==1,np.float32(eto),np.float32(np.nan))
    # sb_old=np.where(mask==1,np.float32(sb_old),np.float32(np.nan))
    # pr=np.where(mask==1,np.float32(pr),np.float32(np.nan))
    # wb_old=np.where(mask==1,np.float32(wb_old),np.float32(np.nan))
    # p=np.where(mask==1,np.float32(p),np.float32(np.nan))
    # tx=np.where(mask==1,np.float32(tx),np.float32(np.nan))
    mask=np.where(classmap==classnum,1,0)
    eto=np.where(mask==1,eto,np.float32(np.nan))
    sb_old=np.where(mask==1,sb_old,np.float32(np.nan))
    pr=np.where(mask==1,pr,np.float32(np.nan))
    wb_old=np.where(mask==1,wb_old,np.float32(np.nan))
    p=np.where(mask==1,p,np.float32(np.nan))
    tx=np.where(mask==1,tx,np.float32(np.nan))

    # kc3=np.where(mask==1,np.float32(kc_list[3]),np.float32(np.nan))
    # kc4=np.where(mask==1,np.float32(kc_list[4]),np.float32(np.nan))
    # xx=np.where((mask==1) & (idoy-istart>=0) & (idoy-iend<=0) & ((idoy-istart)/30.>1.), np.float32(1.), np.float32((idoy-istart)/30.))
    # kc=np.where((idoy-istart>=0) & (idoy-iend<=0), np.float32(kc3*(1.-xx)+(kc4*xx)), np.float32(kc3))
    kc3=np.zeros(mask.shape,dtype='float32')   
    kc3=np.where(mask==1,kc_list[3],kc3)
    kc4=np.zeros(mask.shape,dtype='float32')   
    kc4=np.where(mask==1,kc_list[4],kc4)
    xx=np.where((mask==1) & (idoy-istart>=0) & (idoy-iend<=0) & ((idoy-istart)/30.>1.),1., (idoy-istart)/30.)
    kc=np.where((idoy-istart>=0) & (idoy-iend<=0), kc3*(1.-xx)+(kc4*xx), kc3)
    del kc3,kc4,xx

    etm = kc * eto
    del eto, kc

    # snm=np.where(Fsnm*(tx-tmelt)>sb_old, np.float32(sb_old), np.float32(Fsnm*(tx-tmelt)))
    # snm=np.where(sb_old<=0,np.float32(0),np.float32(snm))
    snm=np.where(Fsnm*(tx-tmelt)>sb_old, sb_old, Fsnm*(tx-tmelt))
    snm=np.where(sb_old<=0,0,snm)
    del tx
    sb=sb_old-snm
    del sb_old

    wb, wx, Eta = eta(mask,wb_old+snm, etm, Sa, D, p, pr)
    del mask,wb_old,snm,p,pr,wx
    # Eta=np.where(Eta>etm,np.float32(etm),np.float32(Eta))
    Eta=np.where(Eta>etm,etm,Eta)

    # return etm, Eta, wb, wx, sb, kc
    # return (etm, Eta, wb, sb)
    # print('warm',mask.dtype,eto.dtype,sb_old.dtype,pr.dtype,wb_old.dtype,p.dtype,tx.dtype,kc.dtype,snm.dtype,etm.dtype,Eta.dtype,wb.dtype,sb.dtype)
    return [etm, Eta, wb, sb]

# @dask.delayed
def EtaCalc_warmest(classmap,classnum,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):
# def EtaCalc_warmest(mask,kc_list,eto,sb_old,pr,wb_old,Sa,D,p,Fsnm,tx,tmelt):
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
    # mask=np.where(classmap==classnum,np.float32(1),np.float32(np.nan))
    # eto=np.where(mask==1,np.float32(eto),np.float32(np.nan))
    # sb_old=np.where(mask==1,np.float32(sb_old),np.float32(np.nan))
    # pr=np.where(mask==1,np.float32(pr),np.float32(np.nan))
    # wb_old=np.where(mask==1,np.float32(wb_old),np.float32(np.nan))
    # p=np.where(mask==1,np.float32(p),np.float32(np.nan))
    # tx=np.where(mask==1,np.float32(tx),np.float32(np.nan))
    mask=np.where(classmap==classnum,1,0)
    eto=np.where(mask==1,eto,np.float32(np.nan))
    sb_old=np.where(mask==1,sb_old,np.float32(np.nan))
    pr=np.where(mask==1,pr,np.float32(np.nan))
    wb_old=np.where(mask==1,wb_old,np.float32(np.nan))
    p=np.where(mask==1,p,np.float32(np.nan))
    tx=np.where(mask==1,tx,np.float32(np.nan))

    # kc=np.where(mask==1,np.float32(kc_list[4]),np.float32(np.nan))
    kc=np.zeros(mask.shape,dtype='float32')   
    kc=np.where(mask==1,kc_list[4],kc) 
    etm = kc * eto
    del eto,kc

    # snm=np.where(Fsnm*(tx-tmelt)>sb_old, np.float32(sb_old), np.float32(Fsnm*(tx-tmelt)))
    # snm=np.where(sb_old<=0,np.float32(0),np.float32(snm))
    snm=np.where(Fsnm*(tx-tmelt)>sb_old, sb_old, Fsnm*(tx-tmelt))
    snm=np.where(sb_old<=0,0,snm)
    del tx
    sb=sb_old-snm
    del sb_old,snm
    
    wb, wx, Eta = eta(mask,wb_old, etm, Sa, D, p, pr)
    del mask,wb_old,p,pr,wx

    # Eta=np.where(Eta>etm,np.float32(etm),np.float32(Eta))
    Eta=np.where(Eta>etm,etm,Eta)
    
    # return etm, Eta, wb, wx, sb, kc
    # return (etm, Eta, wb, sb)
    # print('warmest',mask.dtype,eto.dtype,sb_old.dtype,pr.dtype,wb_old.dtype,p.dtype,tx.dtype,kc.dtype,snm.dtype,etm.dtype,Eta.dtype,wb.dtype,sb.dtype)
    return [etm, Eta, wb, sb]



# def EtaCalc(im_mask,Tx,Ta,Pr,Txsnm,Fsnm,Eto,wb_old,sb_old,istart0,istart1,Sa,D,p,kc_list,lgpt5,eta_class,doy_start,doy_end):
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
    # # we parallelize here by eta_class (the six different regimes for computing ET)
    # # the computations for eta_class are each a delayed function
    # # we call each function which saves the future computation as an object to a list of tasks
    # # then we call compute on the list of tasks at the end to execute them in parallel 
    if parallel:
        import dask

    ncats=5
    # nvars=4
    ETM_list=[]
    ETA_list=[]
    WB_list=[]
    SB_list=[]
    times={}
    compute_times=[]
    agg_times=[]
    start0=timer()
    # results_list=[]
    # results_list
    # print('beginning day loop')
    for idoy in range(doy_start-1, doy_end):
        # compute etm,eta,wb,sb for each eta regime
        # results_list is list of lists: [[etm, eta, wb, sb],[etm, eta, wb, sb],...] in that order
        # may want to use this for when parallel=False, need to pass the parallel flag to LGPCalc

        # vars1=dask.delayed(EtaCalc_snow)(eta_class[:,:,idoy],1,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy])
        # vars2=dask.delayed(EtaCalc_snowmelting)(eta_class[:,:,idoy],2,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm)                                                                       
        # vars3=dask.delayed(EtaCalc_cold)(eta_class[:,:,idoy],3,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm)                                                               
        # vars4=dask.delayed(EtaCalc_warm)(eta_class[:,:,idoy],4,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm,istart0,istart1,idoy)                                        
        # vars5=dask.delayed(EtaCalc_warmest)(eta_class[:,:,idoy],5,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm)
        # task_list=[vars1,vars2,vars3,vars4,vars5]
        # results_list=dask.compute(*task_list) 
        if parallel:      
            start=timer()
            results_list=[]
            eta_class_d=dask.delayed(eta_class[:,:,idoy])
            Eto_d=dask.delayed(Eto[:,:,idoy])
            sb_old_d=dask.delayed(sb_old)
            Pr_d=dask.delayed(Pr[:,:,idoy])
            wb_old_d=dask.delayed(wb_old)
            p_d=dask.delayed(p[:,:,idoy])
            Tx_d=dask.delayed(Tx[:,:,idoy])
            vars1=dask.delayed(EtaCalc_snow)(eta_class_d,1,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d)
            vars2=dask.delayed(EtaCalc_snowmelting)(eta_class_d,2,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d,Fsnm,Tx_d,Txsnm)                                                                       
            vars3=dask.delayed(EtaCalc_cold)(eta_class_d,3,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d,Fsnm,Tx_d,Txsnm)                                                               
            vars4=dask.delayed(EtaCalc_warm)(eta_class_d,4,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d,Fsnm,Tx_d,Txsnm,istart0,istart1,idoy)                                        
            vars5=dask.delayed(EtaCalc_warmest)(eta_class_d,5,kc_list,Eto_d,sb_old_d,Pr_d,wb_old_d,Sa,D,p_d,Fsnm,Tx_d,Txsnm)
            task_list=[vars1,vars2,vars3,vars4,vars5]
            results_list=dask.compute(*task_list)
            compute_times.append(timer()-start)

            # aggregate results for each variable
            start=timer()
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
            agg_times.append(timer()-start)            

        else:
            start=timer()
            results_list=[]            
            vars1=EtaCalc_snow(eta_class[:,:,idoy],1,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy])
            vars2=EtaCalc_snowmelting(eta_class[:,:,idoy],2,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm)                                                                       
            vars3=EtaCalc_cold(eta_class[:,:,idoy],3,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm)                                                               
            vars4=EtaCalc_warm(eta_class[:,:,idoy],4,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm,istart0,istart1,idoy)                                        
            vars5=EtaCalc_warmest(eta_class[:,:,idoy],5,kc_list,Eto[:,:,idoy],sb_old,Pr[:,:,idoy],wb_old,Sa,D,p[:,:,idoy],Fsnm,Tx[:,:,idoy],Txsnm)
            results_list.append([vars1,vars2,vars3,vars4,vars5])            
            compute_times.append(timer()-start)

            # aggregate results for each variable
            start=timer()
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
            agg_times.append(timer()-start)
        
        ETM_list.append(ETM_agg)
        ETA_list.append(ETA_agg)
        WB_list.append(WB_agg)
        SB_list.append(SB_agg)

        wb_old=WB_agg.copy()
        sb_old=SB_agg.copy()
    task_time0=timer()-start0

    times['chunk_size']=Tx.shape
    times['complete day loop']=task_time0   
    times['avg daily compute'] = np.array(compute_times).mean()
    times['avg daily agg'] = np.array(agg_times).mean()
    del im_mask,Tx,Pr,Txsnm,Fsnm,Eto,wb_old,sb_old,istart0,istart1,Sa,D,p,kc_list,lgpt5,eta_class
    del vars1,vars2,vars3,vars4,vars5,results_list,ETM_agg,ETA_agg,WB_agg,SB_agg

    start=timer()
    ETM=np.stack(ETM_list,axis=-1,dtype='float32') # stack on last dim (time)
    ETA=np.stack(ETA_list,axis=-1,dtype='float32') # stack on last dim (time)
    # print('ETM,ETA info',ETM.shape,ETM.dtype,ETA.shape,ETA.dtype)
    task_time=timer()-start
    times['stack']=task_time
    # print('ETM,ETA1',ETM.dtype,ETA.dtype)     
    del ETM_list,ETA_list

    start=timer()
    ETA=np.where(ETA<0,0,ETA)
    task_time=timer()-start
    times['ETA where']=task_time  
    # print('ETA2',ETA.dtype)

    start=timer()
    with np.errstate(invalid='ignore'):  #KLG
        ETMx = np.append(ETM, ETM[:,:,0:12],axis=2)  
        ETAx = np.append(ETA, ETA[:,:,0:12],axis=2)  
    # print('ETMx,ETAx info',ETMx.shape,ETMx.dtype,ETAx.shape,ETAx.dtype)
    task_time=timer()-start
    times['append']=task_time  
    # print('ETMx,ETAx',ETMx.dtype,ETAx.dtype)
    del ETM, ETA
    # print(ETAx.shape,ETMx.shape)

    start=timer()
    xx = val10day(ETAx)  #KLG
    yy = val10day(ETMx)  #KLG
    task_time=timer()-start
    times['xx,yy']=task_time 
    # print('xx,yy',xx.dtype,yy.dtype)
    del ETMx,ETAx

    start=timer()
    with np.errstate(divide='ignore', invalid='ignore'):  #KLG
        lgp_whole = xx[:,:,:doy_end]/yy[:,:,:doy_end]  #KLG
    task_time=timer()-start
    times['lgp_whole']=task_time 
    del xx,yy
    # print('lgp_whole',lgp_whole.dtype)

    start=timer()
    lgp_tot=np.where((islgp==1)&(lgp_whole>=0.4),1,0).sum(axis=2)  #KLG
    task_time=timer()-start
    times['lgp_tot']=task_time   
    del lgp_whole   
    # print('lgp_tot',lgp_tot.dtype)
 
    # print(times)   
    return lgp_tot
    # return ETAx     

    
    # now compute everything
    # result_list is a list of tuples containing arrays: [(etm,eta_var,wb,wx,sb,kc from EtaCalc_snow),(etm,eta_var,wb,wx,sb,kc from EtaCalc_snowmelting),(etm,eta_var,wb,wx,sb,kc from EtaCalc_cold),...]
    # the tuples are returned in the order the functions were called
    # to access a single array in the result list we use two sets of brackets
    # e.g. result_list[1][2] is wb from the EtaCalc_snowmelting function
    # print('in LGPCalc before parallel compute', psutil.virtual_memory().free/1E9)
    # print('in LGPCalc, computing ET components for day',idoy+1)


# @jit(nopython=True)
# def setdat(dat1):
#     if dat1 > 365:
#         dat1 = dat1-365
#     return dat1


# @jit(nopython=True)
# def islgpt(Ta):
#     ist5 = np.zeros((np.shape(Ta)))
#     for i in range(len(Ta)):
#         if Ta[i] >= 5:
#             ist5[i] = 1
#         else:
#             ist5[i] = 0

#     return ist5

