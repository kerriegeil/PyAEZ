"""
PyAEZ version 2.1.0 (June 2023)
Additional calculations used throughout AEZ modules
2020: N. Lakmal Deshapriya
2022/2023: Swun Wunna Htet, K. Boonma
"""

import numpy as np
from scipy.interpolate import interp1d
try:
    import gdal
except:
    from osgeo import gdal
import dask.array as da
import dask
from collections import OrderedDict

class UtilitiesCalc(object):
    def __init__(self,chunk2D=None,chunk3D=None):
        """Initiate a Utilities Class instance

        Args:
            chunk2D (len 2 tuple of int): chunk size for 2D arrays
            chunk2D (len 3 tuple of int): chunk size for 3D arrays
        """        
        self.chunk2D = chunk2D
        self.chunk3D = chunk3D  

    def interpMonthlyToDaily(self, monthly_vector, cycle_begin, cycle_end, no_minus_values=False):
        """Interpolate monthly climate data to daily climate data

        Args:
            monthly_vector (1D NumPy): monthly data that needs interpolating to daily 
            cycle_begin (int): Starting Julian day
            cycle_end (int): Ending Julian day
            no_minus_values (bool, optional): Set minus values to zero. Defaults to False.

        Returns:
            1D NumPy: Daily climate data vector
        """        

        doy_middle_of_month = np.arange(0,12)*30 + 15 # Calculate doy of middle of month

        #daily_vector = np.interp(np.arange(cycle_begin,cycle_end+1), doy_middle_of_month, monthly_vector)

        inter_fun = interp1d(doy_middle_of_month, monthly_vector, kind='quadratic', fill_value='extrapolate')
        daily_vector = inter_fun( np.arange(cycle_begin,cycle_end+1) )

        if no_minus_values:
            daily_vector[daily_vector<0] = 0

        return daily_vector

    def averageDailyToMonthly(self, daily_arr):
    # def averageDailyToMonthly(self, daily_vector):
        """Aggregating daily data into monthly data

        Args:
            daily_arr (3D NumPy): daily data array of dims (ny,nx,365)

        Returns:
            3D NumPy: Monthly data array of dims (ny,nx,12)
        """        
        # monthly_vector = np.zeros(12)

        # monthly_vector[0] = np.sum(daily_vector[:31])/31
        # monthly_vector[1] = np.sum(daily_vector[31:59])/28
        # monthly_vector[2] = np.sum(daily_vector[59:90])/31
        # monthly_vector[3] = np.sum(daily_vector[90:120])/30
        # monthly_vector[4] = np.sum(daily_vector[120:151])/31
        # monthly_vector[5] = np.sum(daily_vector[151:181])/30
        # monthly_vector[6] = np.sum(daily_vector[181:212])/31
        # monthly_vector[7] = np.sum(daily_vector[212:243])/31
        # monthly_vector[8] = np.sum(daily_vector[243:273])/30
        # monthly_vector[9] = np.sum(daily_vector[273:304])/31
        # monthly_vector[10] = np.sum(daily_vector[304:334])/30
        # monthly_vector[11] = np.sum(daily_vector[334:])/31

        # delay the input data so it's copied once instead of at each call of the function
        daily_arr=dask.delayed(daily_arr)

        # a nested ordered dictionary containing info for each month
        month_info = OrderedDict({ 'Jan': {'ndays':31,'lim_lo':0,'lim_hi':31},
                            'Feb':{'ndays':28,'lim_lo':31,'lim_hi':59},
                            'Mar':{'ndays':31,'lim_lo':59,'lim_hi':90},
                            'Apr':{'ndays':30,'lim_lo':90,'lim_hi':120},
                            'May':{'ndays':31,'lim_lo':120,'lim_hi':151},
                            'Jun':{'ndays':30,'lim_lo':151,'lim_hi':181},
                            'Jul':{'ndays':31,'lim_lo':181,'lim_hi':212},
                            'Aug':{'ndays':31,'lim_lo':212,'lim_hi':243},
                            'Sep':{'ndays':30,'lim_lo':243,'lim_hi':273},
                            'Oct':{'ndays':31,'lim_lo':273,'lim_hi':304},
                            'Nov':{'ndays':30,'lim_lo':304,'lim_hi':334},
                            'Dec':{'ndays':31,'lim_lo':334,'lim_hi':365} })

        # put the computations inside a delayed function
        @dask.delayed
        def monthly_aggregate(daily,ndays,lim_lo,lim_hi):
            monthly=daily[:,:,lim_lo:lim_hi].sum(axis=2)/ndays
            return monthly

        # in a regular non-delayed loop, call delayed function and compile list of future compute tasks
        task_list=[]                        
        for month_inputs in month_info.values():
            task=monthly_aggregate(daily_arr,month_inputs['ndays'],month_inputs['lim_lo'],month_inputs['lim_hi'])
            task_list.append(task)

        # compute tasks in parallel
        # this returns a list of arrays in the same order as month_info
        data_list=dask.compute(*task_list)

        # stack the results along a 3rd dimension (ny,nx,12)
        monthly_arr=np.stack(data_list,axis=-1,dtype='float32')        

        # monthly_vector = da.zeros((daily_vector.shape[0],daily_vector.shape[1],12),chunks=self.chunk3D)
        # monthly_vector = np.zeros((daily_vector.shape[0],daily_vector.shape[1],12))

        # monthly_vector[:,:,0] = daily_vector[:,:,:31].sum(axis=2)/31
        # monthly_vector[:,:,1] = daily_vector[:,:,31:59].sum(axis=2)/28
        # monthly_vector[:,:,2] = daily_vector[:,:,59:90].sum(axis=2)/31
        # monthly_vector[:,:,3] = daily_vector[:,:,90:120].sum(axis=2)/30
        # monthly_vector[:,:,4] = daily_vector[:,:,120:151].sum(axis=2)/31
        # monthly_vector[:,:,5] = daily_vector[:,:,151:181].sum(axis=2)/30
        # monthly_vector[:,:,6] = daily_vector[:,:,181:212].sum(axis=2)/31
        # monthly_vector[:,:,7] = daily_vector[:,:,212:243].sum(axis=2)/31
        # monthly_vector[:,:,8] = daily_vector[:,:,243:273].sum(axis=2)/30
        # monthly_vector[:,:,9] = daily_vector[:,:,273:304].sum(axis=2)/31
        # monthly_vector[:,:,10] = daily_vector[:,:,304:334].sum(axis=2)/30
        # monthly_vector[:,:,11] = daily_vector[:,:,334:].sum(axis=2)/31            

        return monthly_arr
        # return monthly_vector.compute()

    def generateLatitudeMap(self, lat_min, lat_max, location, im_height, im_width, chunk2D):  #KLG
    # def generateLatitudeMap(self, lat_min, lat_max, im_height, im_width):  
    # def generateLatitudeMap(self, lats, location):  #KLG

        """Create latitude map from input geographical extents

        Args:
            lat_min (float): the minimum latitude
            lat_max (float): the maximum latitude
            im_height (float): height of the input raster (pixels,grid cells)
            im_width (float): width of the input raster (pixels,grid cells)

        Returns:
            2D NumPy: interpolated 2D latitude map 
        """        
        # lat_step=(lat_max-lat_min)/im_height  
        # lat_lim = np.linspace(lat_min+lat_step/2, lat_max-lat_step/2, im_height)  
        # lon_lim = np.linspace(1, 1, im_width) # just temporary lon values, will not affect output of this function.  
        # [X_map,Y_map] = np.meshgrid(lon_lim,lat_lim)  
        # lat_map = np.flipud(Y_map) 

        # Generate a 2D array of latitude  #KLG
        # for parallel computing
        if chunk2D:
            # For lat_min, lat_max values given at pixel centers  #KLG
            if location:
                lat_vals=da.linspace(lat_min, lat_max, im_height).astype('float32')  #KLG
                lat_map=da.tile(lat_vals[:,np.newaxis],(1,im_width)).rechunk(chunks=chunk2D)  #KLG            
            # For lat_min, lat_max values given at exterior pixel edges  #KLG
            if ~location:
                lat_step=(lat_max-lat_min)/im_height  #KLG
                lat_vals = da.linspace(lat_min+lat_step/2, lat_max-lat_step/2, im_height)  #KLG
                lat_map=da.tile(lat_vals[:,np.newaxis],(1,im_width)).rechunk(chunks=chunk2D)  #KLG     
        # for serial computing
        else:     
            if location:
                lat_vals=np.linspace(lat_min, lat_max, im_height).astype('float32')  #KLG
                lat_map=np.tile(lat_vals[:,np.newaxis],(1,im_width))  #KLG
            if ~location:
                lat_step=(lat_max-lat_min)/im_height  #KLG
                lat_vals = np.linspace(lat_min+lat_step/2, lat_max-lat_step/2, im_height)  #KLG
                lat_map=np.tile(lat_vals[:,np.newaxis],(1,im_width))  #KLG                

        # precision issues can arise from above, why not just take a 1D lat array of pixel centers as an input  #KLG
        # instead of recreating them with linspace. Then, use below to make the 2D array  #KLG
        # lat_map=np.tile(lats[:,np.newaxis],(1,im_width))  #KLG
        return lat_map

    def classifyFinalYield(self, est_yield):

        ''' Classifying Final Yield Map
        class 5 = very suitable = yields are equivalent to 80% or more of the overall maximum yield,
        class 4 = suitable = yields between 60% and 80%,
        class 3 = moderately suitable = yields between 40% and 60%,
        class 2 = marginally suitable = yields between 20% and 40%,
        class 1 = not suitable = yields between 0% and 20%.
        '''

        est_yield_max = np.amax( est_yield[est_yield>0] )
        est_yield_min = np.amin( est_yield[est_yield>0] )

        est_yield_20P = (est_yield_max-est_yield_min)*(20/100) + est_yield_min
        est_yield_40P = (est_yield_max-est_yield_min)*(40/100) + est_yield_min
        est_yield_60P = (est_yield_max-est_yield_min)*(60/100) + est_yield_min
        est_yield_80P = (est_yield_max-est_yield_min)*(80/100) + est_yield_min

        est_yield_class = np.zeros(est_yield.shape)

        est_yield_class[ np.all([0<est_yield, est_yield<=est_yield_20P], axis=0) ] = 1 # not suitable
        est_yield_class[ np.all([est_yield_20P<est_yield, est_yield<=est_yield_40P], axis=0) ] = 2 # marginally suitable
        est_yield_class[ np.all([est_yield_40P<est_yield, est_yield<=est_yield_60P], axis=0) ] = 3 # moderately suitable
        est_yield_class[ np.all([est_yield_60P<est_yield, est_yield<=est_yield_80P], axis=0) ] = 4 # suitable
        est_yield_class[ np.all([est_yield_80P<est_yield], axis=0)] = 5 # very suitable

        return est_yield_class

    def saveRaster(self, ref_raster_path, out_path, numpy_raster):
        """Save NumPy arrays/matrices to GeoTIFF files

        Args:
            ref_raster_path (string): File path to referece GeoTIFF for geo-tagged info.
            out_path (string): Path for the created GeoTIFF to be saved as/to
            numpy_raster (2D NumPy): the arrays to be saveda as GeoTIFF
        """        
        # Read random image to get projection data
        img = gdal.Open(ref_raster_path)
        # allocating space in hard drive
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(out_path, img.RasterXSize, img.RasterYSize, 1, gdal.GDT_Float32)
        # set image paramenters (imfrormation related to cordinates)
        outdata.SetGeoTransform(img.GetGeoTransform())
        outdata.SetProjection(img.GetProjection())
        # write numpy matrix as new band and set no data value for the band
        outdata.GetRasterBand(1).WriteArray(numpy_raster)
        outdata.GetRasterBand(1).SetNoDataValue(-999)
        # flush data from memory to hard drive
        outdata.FlushCache()
        outdata=None

    def averageRasters(self, raster_3d):
        """Averaging a list of raster files in time dimension

        Args:
            raster_3d (3D NumPy array): any climate data

        Returns:
            2D NumPy: the averaged climate data into 'one year' array
        """        
        # input should be a 3D raster and averaging will be done through last dimension (usually corresponding to years)
        return np.sum(raster_3d, axis=2)/raster_3d.shape[-1]

    def windSpeedAt2m(self, wind_speed, altitude):
        """Convert windspeed at any altitude to those at 2m altitude

        Args:
            wind_speed (1D,2D,or 3D NumPy array): wind speed
            altitude (float): altitude [m]

        Returns:
            1D,2D,or 3D NumPy array: Converted wind speed at 2m altitude
        """        
        # this function converts wind speed from a particular altitude to wind speed at 2m altitude. wind_speed can be a numpy array (can be 1D, 2D or 3D)
        return wind_speed * (4.87/np.log(67.8*altitude-5.42))


    def smoothDailyTemp(self, day_start, day_end, mask, daily_T, chunk3D):  #KLG
        """create smoothed daily temperature curve using 5th degree spline 

        Args:
            day_start (scalar): first day of the data in Julian day 
            day_end (scalar): last day of the data in Julian day 
            mask (2D integer array): administrative mask of 0's and 1's, if user doesn't create this it comes in as all 1's
            daily_T (3D array): daily temperature

        Returns:
            3D NumPy array: 5th degree spline smoothed temperature
        """          

        if chunk3D:
            self.chunk3D=chunk3D
            self.parallel=True

        if self.parallel:
            # spline fitting
            def polyfit_polyval(x,y,deg):
                coefs=np.polynomial.polynomial.polyfit(x,y,deg=deg)         
                spline=np.polynomial.polynomial.polyval(x,coefs) 
                return spline.astype('float32') 

            deg=5  # polynomial degree for spline fitting
            days = np.arange(day_start,day_end+1) # x values  #KLG

            # replace any nan in the data with zero (nans may be present under a mask) #KLG
            mask3D = da.tile(mask[:,:,np.newaxis], (1,1,days.shape[0])).rechunk(chunks=self.chunk3D)  #KLG
            data=da.where(mask3D==0,0,daily_T)   #KLG

            # collapse lat and lon because polyfit and polyval only work on data up to 2 dimensions
            data2D=data.transpose(2,0,1).reshape(days.shape[0],-1).rechunk({0:-1,1:'auto'})

            # delay data so it's only passed to computations once
            days=dask.delayed(days)
            delayed_chunks=data2D.to_delayed()
            
            task_list = [dask.delayed(polyfit_polyval)(days,dchunk,deg) for dchunk in delayed_chunks[0,:]]
            results_list=dask.compute(*task_list)

            interp_daily_temp=np.concatenate(results_list).reshape(mask3D.shape[0],mask3D.shape[1],-1)  #KLG

        else:
            days = np.arange(day_start,day_end+1) # x values  #KLG

            # replace any nan (i.e. if there is a mask) in the data with zero  #KLG
            mask3D = np.tile(mask[:,:,np.newaxis], (1,1,days.shape[0]))  #KLG
            data=np.where(mask3D==0,0,daily_T)   #KLG
            data2D=data.transpose(2,0,1).reshape(days.shape[0],-1) # every column is a set of y values  #KLG
            del data  #KLG

            # do the spline fitting  #KLG
            quad_spl=np.polynomial.polynomial.polyfit(days,data2D,deg=5)  #KLG
            interp_daily_temp=np.polynomial.polynomial.polyval(days,quad_spl)  #KLG

            #reshape  #KLG
            interp_daily_temp=interp_daily_temp.reshape(mask3D.shape[0],mask3D.shape[1],-1)  #KLG
            interp_daily_temp=np.where(mask3D==0,np.nan,interp_daily_temp)
        
        return interp_daily_temp.astype('float32')   #KLG

    def setChunks(self,nchunks,nlons):

        # how many longitudes per chunk
        chunk_nlons=int(da.ceil(nlons/nchunks))

        # dimensions of a single chunk for 3D and 2D arrays, -1 means all latitudes and all times
        chunk3D=(-1,chunk_nlons,-1)
        chunk2D=(-1,chunk_nlons)

        return chunk2D, chunk3D

