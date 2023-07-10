
"""
PyAEZ: LGPCalc.py calculates the length of growing period (LGP)
2022: K.Boonma ref. GAEZv4
"""

import numba as nb
import numpy as np


# @nb.jit(nopython=True)
def rainPeak(totalPrec_monthly, meanT_daily, lgpt5_point):
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
#     # THIS IS NOT WORKING
#     days_f = np.arange(0., 365)
#     lgpt5_veg = days_f[meanT_daily >= 5.0]
#     if lgpt5_point < 365.:
#         istart0 = lgpt5_veg[0]
#         istart1 = setdat(istart0) + lgpt5_point-1
#     else:
#         istart0 = 0.
#         istart1 = lgpt5_point-1

    ndays=meanT_daily.shape[0]
    # get index of first occurrence in time where true at each grid cell
    day1=np.argmax(meanT_daily>=5.0,axis=0) 
    # argmax returns 0 where there is no first occurrence (no growing season) so need to fix
    if lgpt5_point==0: day1=np.nan 

    if lgpt5_point>ndays: 
        istart0=0
    else:
        istart0=day1

    if istart0>ndays: 
        dat1=istart0-ndays
    else:
        dat1=istart0

    if lgpt5_point<ndays: 
        istart1=dat1+lgpt5_point-1
    else:
        istart1=lgpt5_point-1

    return meanT_daily, istart0, istart1
# ============================================


@nb.jit(nopython=True)
def isfromt0(meanT_daily_new, doy):
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

# ============================================


@nb.jit(nopython=True)
def eta(wb_old, etm, Sa, D, p, rain):
    """SUBROUTINE: Calculate actual evapotranspiration (ETa) 

    Args:
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
    wx = 0
    Salim = max(Sa*D, 1.)
    wr=min(100*(1-p),Salim)

    if rain >= etm:
        eta = etm
    # elif rain+wb_old-wr >= etm:
    elif s-wr >= etm:
        eta = etm
    else:
        rho = wb_old/wr
        eta = min(rain + rho*etm, etm)
        # eta = rain+rho*etm


    wb=s-eta
    # wb = min(wb_old+rain-eta, Salim)
    # wb = wb - eta

    if wb >= Salim:
        wx = wb-Salim
        wb = Salim
    else:
        wx = 0

    
    if wb < 0:
        wb=0

    return wb, wx, eta


@nb.jit(nopython=True)
def psh(ng, et0):
    """Calculate soil moisture depletion fraction (0-1)

    Args:
        ng (float): crop group
        et0 (float): potential evapotranspiration [mm/day]

    Returns:
        float: soil moisture depletion fraction
    """
    # ng = crop group
    # eto = potential evapotranspiration [mm/day]
    if ng == 0.:
        psh0 = 0.5
    else:
        psh0 = 0.3+(ng-1)*.05

    psh = psh0 + .04 * (5.-et0)

    if psh < 0.1:
        psh = 0.1
    elif psh > 0.8:
        psh = 0.8

    return psh


@nb.jit(nopython=True)
def val10day(Et):
    """Calculate 10-day moving average 

    Args:
        
    """
    # Program to calculate moving average
    arr = Et
    window_size = 10
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i: i + window_size]
        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        # Shift window to right by one position
        i += 1

    return np.array(moving_averages)


@nb.jit(nopython=True)
def EtaCalc(Tx365, Ta365, Pcp365, Txsnm, Fsnm, Eto365, wb_old, sb_old, doy, istart0, istart1, Sa, D, p, kc_list, lgpt5_point):
    """Calculate actual evapotranspiration (ETa)"""

    # Period with Tmax <= Txsnm (precipitaton falls as snow as is added to snow bucket)
    if Tx365 <= Txsnm and Ta365 <= 0.:
        etm = kc_list[0] * Eto365
        Etm365 = etm
        # ks = 0.
        # snm = min(Fsnm*(Tx365-Txsnm), sb_old)  # this case has no snow-melt
        sbx = sb_old+Pcp365
        
        wb, wx, Eta = eta(wb_old-Pcp365, etm, Sa, D, p, Pcp365)

        Salim = Sa*D  

        if sbx >= etm:
            Sb365 = sbx-etm
            Eta365 = etm

            wb=wb_old-etm
            if wb > Salim:
                wx = wb- Salim 
                wb = Salim
            else:
                wx = 0
        else:
            Sb365 = 0.
            Eta365 = Eta

        if wb < 0:
            wb = 0

        Wb365 = wb
        Wx365 = wx
        kc365 = kc_list[0]

    # Snow-melt takes place; minor evapotranspiration
    elif Ta365 <= 0. and Tx365 >= 0.:
        etm = kc_list[1] * Eto365
        Etm365 = etm
        ks = 0.1
        # Snow-melt function
        snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        sbx = sb_old - snm #- (ks*etm)
        Salim = Sa*D
        if sbx >= etm:
            Sb365 = sbx-etm
            Eta365 = etm
            wb = wb_old+snm+Pcp365-etm

            if wb > Salim:
                wx = wb - Salim
                wb = Salim
            else:
                wx = 0
        else:
            Sb365 = 0.
            wb, wx, Eta = eta(wb_old+snm, etm, Sa, D, p, Pcp365)
            Eta365 = Eta

        if wb < 0:
            wb = 0
      

            # wb=wb0+snm
        # if Eta365 > Etm365:
        #     Eta365 = Etm365
        Wb365 = wb
        Wx365 = wx
        kc365 = kc_list[1]

    elif Ta365 < 5. and Ta365 > 0.:
        # Biological activities before start of growing period
        etm = kc_list[2] * Eto365
        Etm365 = etm
        # ks = 0.2
        # In case there is still snow
        if sb_old > 0.:
            snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        else:
            snm = 0.

        sbx = sb_old-snm

        wb, wx, Eta = eta(wb_old+snm, etm, Sa, D, p, Pcp365)

        if Eta > Etm365:
            Eta365 = etm
        else:
            Eta365=Eta

        Wx365 = wx
        Wb365 = wb
        Sb365 = sbx
        kc365 = kc_list[2]

    elif lgpt5_point < 365 and Ta365 >= 5.:
        if doy >= istart0 and doy <= istart1:
            # case 2 -- kc increases from 0.5 to 1.0 during first month of LGP
            # case 3 -- kc = 1 until daily Ta falls below 5C
            xx = min((doy-istart0)/30., 1.)
            kc = kc_list[3]*(1.-xx)+(kc_list[4]*xx)
        else:
            # case 1 -- kc=0.5 for days until start of growing period
            kc = kc_list[3]

        etm = kc * Eto365
        Etm365 = etm
        # In case there is still snow
        if sb_old > 0.:
            snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        else:
            snm = 0.

        sbx = sb_old-snm

        wb, wx, Eta = eta(wb_old+snm, etm, Sa, D, p, Pcp365)

        if Eta > Etm365:
            Eta365 = etm
        else:
            Eta365 = Eta

        Wb365 = wb
        Wx365 = wx
        Sb365 = sbx
        kc365 = kc

    else:
        etm = kc_list[4] * Eto365
        Etm365 = etm
        # In case there is still snow
        if sb_old > 0.:
            snm = min(Fsnm*(Tx365-Txsnm), sb_old)
        else:
            snm = 0.

        # sbx = sb_old-snm

        wb, wx, Eta = eta(wb_old, etm, Sa, D, p, Pcp365)

        if Eta > Etm365:
            Eta365 = etm
        else:
            Eta365 = Eta
            
        Wb365 = wb
        Wx365 = wx
        Sb365 = sbx
        kc365 = kc_list[4]

    return Eta365, Etm365, Wb365, Wx365, Sb365, kc365


@nb.jit(nopython=True)
def islgpt(Ta):
    ist5 = np.zeros((np.shape(Ta)))
    for i in range(len(Ta)):
        if Ta[i] >= 5:
            ist5[i] = 1
        else:
            ist5[i] = 0

    return ist5


@nb.jit(nopython=True)
def lendat(dat1, dat2):
    if dat1 <= dat2:
        lendat = dat2-dat1+1
    else:
        lendat = (365+dat2)-dat1+1

    return lendat


@nb.jit(nopython=True)
def setdat(dat1):
    if dat1 > 365:
        dat1 = dat1-365
    return dat1



