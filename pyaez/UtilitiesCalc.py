"""
PyAEZ version 2.2 (4 JAN 2024)
Additional calculations used throughout AEZ modules
2020: N. Lakmal Deshapriya
2022/2023: Swun Wunna Htet, K. Boonma
2023 (Dec): Swun Wunna Htet
2024: Kerrie Geil (vectorize and parallelize with dask)

Modification:
1. Latitude calculated revised according to GAEZ Fortran routine.
"""

import numpy as np
from scipy.interpolate import interp1d
try:
    import gdal
except:
    from osgeo import gdal
from collections import OrderedDict
import psutil

class UtilitiesCalc(object):
    def __init__(self,chunk2D=None,chunk3D=None):
        """Initiate a Utilities Class instance with chunk 
            dimensions and parallelization flag attached

        Args:
            chunk2D (len 2 tuple of int): the dimensions of a single chunk of a 2D data array
            chunk2D (len 3 tuple of int): the dimensions of a single chunk of a 3D data array
        """        
        self.chunk2D = chunk2D
        self.chunk3D = chunk3D
        if self.chunk2D and self.chunk3D:  
            self.parallel=True
        else:
            self.parallel=False

    def setChunks(self,nchunks,shape,reduce_mem_used,ram,threads):
        """
        Computes an appropriate chunk size based on the user's 
        available computer resources (RAM and processing threads)

        Args:
            nchunks (integer): user override for total number of chunks. Use for debugging only. 
                Defaults to None here and then nchunks is set based on the user's computer resources
                in UtilitiesCalc.setChunks
            shape (length 3 tuple): shape of the 3-dimensional input data arrays (e.g min_temp, max_temp, precipitation, etc.)
            reduce_mem_used (boolean): user option to reduce the chunk size by a factor of 2, which
                reduces RAM usage. Use for debugging only. Defaults to False.

        ---
        Returns:
            chunk2D (length 2 tuple): the dimensions of a single chunk of a 2D data array
            chunk3D (length 3 tuple): the dimensions of a single chunk of a 3D data array
            chunksize3D_MB (float): the approximate size in MB of a single chunk of a 3D data array
            nchunks (integer): total number of chunks (along the longitude dimension) per data array
        """          
        nlats=shape[0]
        nlons=shape[1]
        ntimes=shape[2]

        func_scale_factor=20  # estimated based on RAM usage of setDailyClimateData (the biggest RAM hog)
        dask_scale_factor=2  # dask likely stores at least two chunks per thread
        buff=0#.25E9 # RAM buffer if needed in the future

        # the following should eventually be scaled to a certain size of required RAM
        # e.g. the multiplier (currently 2) should mean that RAM usage is kept under xGB
        if reduce_mem_used: func_scale_factor=func_scale_factor*2        

        if nchunks:
            # user override for default nchunks
            nlons_chunk=int(np.ceil(nlons/nchunks)) # how many longitudes per chunk
        else:
            if (ram>0) and (threads>0):
                ram=ram*1e9  # GB to bytes
                RAMperthread=ram/threads
            else:
                # default nchunks based on system properties    
                RAMinfo=psutil.virtual_memory() # returns info about system RAM in bytes
                threads=psutil.cpu_count()  # returns system number of threads            
                RAMperthread = (RAMinfo.free-buff)/threads
            
            chunklim=RAMperthread/func_scale_factor/dask_scale_factor # biggest chunk size in GB where computations won't fail
            npoints=chunklim/4

            # we chunk only by longitude, so npoints must contain all lats and all times
            nlons_chunk=int(np.floor(npoints/nlats/ntimes)) # number of longitudes per chunk 
            if nlons_chunk<1:
                sys.exit('This computer does not have enough RAM to run PyAEZ with the given inputs. Try using a model domain that is smaller in the latitude dimension.')
            nchunks=int(np.ceil(nlons/nlons_chunk))

            # making sure we have at least as many chunks as threads
            # for machines with many threads sometimes nchunks will still be 1 less than nthreads (for small domains)
            if nchunks < threads:
                nlons_chunk=int(np.ceil(nlons/threads))
                nchunks=int(np.ceil(nlons/nlons_chunk))

        # dimensions of a single chunk for 3D and 2D arrays, -1 means all latitudes or all times
        chunk3D=(-1,nlons_chunk,-1)
        chunk2D=(-1,nlons_chunk)

        # approximate size in MB of a 3D array chunk
        chunksize3D_MB=nlats*nlons_chunk*ntimes*4/1E6

        return chunk2D, chunk3D, chunksize3D_MB, nchunks


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
        #####################################################################
        ##### THIS FUNCTION NOT YET UPDATED FOR VECTORIZED/PARALLELIZED #####
        #####################################################################

        doy_middle_of_month = np.arange(0,12)*30 + 15 # Calculate doy of middle of month

        #daily_vector = np.interp(np.arange(cycle_begin,cycle_end+1), doy_middle_of_month, monthly_vector)

        inter_fun = interp1d(doy_middle_of_month, monthly_vector, kind='quadratic', fill_value='extrapolate')
        daily_vector = inter_fun( np.arange(cycle_begin,cycle_end+1) )

        if no_minus_values:
            daily_vector[daily_vector<0] = 0

        return daily_vector

    def averageDailyToMonthly(self,daily_arr):
    # def averageDailyToMonthly(self, daily_vector):
        """Aggregating daily data into monthly data

        Args:
            daily_arr (3D NumPy): daily data array of dims (ny,nx,365)

        Returns:
            3D NumPy: Monthly data array of dims (ny,nx,12)
        """        
        if self.parallel:
            import dask
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
                # with np.errstate(invalid='ignore',divide='ignore'):
                monthly=np.nansum(daily[:,:,lim_lo:lim_hi],axis=2,dtype='float32')/ndays
                return monthly

            # in a regular non-delayed loop, call delayed function and compile list of future compute tasks
            task_list=[]                        
            for month_inputs in month_info.values():
                task=monthly_aggregate(daily_arr,month_inputs['ndays'],month_inputs['lim_lo'],month_inputs['lim_hi'])
                task_list.append(task)

            # compute tasks in parallel
            data_list=dask.compute(*task_list) # a list of arrays in the same order as month_info

            # stack the results along a 3rd dimension (ny,nx,12)
            monthly_arr=np.stack(data_list,axis=-1,dtype='float32')  
        else:      

            monthly_arr = np.zeros((daily_arr.shape[0],daily_arr.shape[1],12),dtype='float32')

            monthly_arr[:,:,0] = daily_arr[:,:,:31].sum(axis=2)/31
            monthly_arr[:,:,1] = daily_arr[:,:,31:59].sum(axis=2)/28
            monthly_arr[:,:,2] = daily_arr[:,:,59:90].sum(axis=2)/31
            monthly_arr[:,:,3] = daily_arr[:,:,90:120].sum(axis=2)/30
            monthly_arr[:,:,4] = daily_arr[:,:,120:151].sum(axis=2)/31
            monthly_arr[:,:,5] = daily_arr[:,:,151:181].sum(axis=2)/30
            monthly_arr[:,:,6] = daily_arr[:,:,181:212].sum(axis=2)/31
            monthly_arr[:,:,7] = daily_arr[:,:,212:243].sum(axis=2)/31
            monthly_arr[:,:,8] = daily_arr[:,:,243:273].sum(axis=2)/30
            monthly_arr[:,:,9] = daily_arr[:,:,273:304].sum(axis=2)/31
            monthly_arr[:,:,10] = daily_arr[:,:,304:334].sum(axis=2)/30
            monthly_arr[:,:,11] = daily_arr[:,:,334:].sum(axis=2)/31            

        return monthly_arr

    def generateLatitudeMap(self, lat_min, lat_max, location, im_height, im_width):    
    # def generateLatitudeMap(self, lats, location):    
        # precision issues can arise from recalculating lat/lon, why not just take a 1D lat array of 
        # pixel centers as an input instead of recreating them with linspace? Then, use this to make 
        # the 2D array:  lat_map=np.broadcast_to(lats[:,np.newaxis],(im_height,im_width))  
        """Create latitude map from input geographical extents

        Args:
            lat_min (float): the minimum latitude at either a grid cell center or grid exterior edge
            lat_max (float): the maximum latitude at either a grid cell center or grid exterior edge
            location (boolean): True = lat_min and lat_max values are located at the center of a grid cell. 
                False = lat_min and lat_max values are located at the exterior edge of a grid cell.
            im_height (float): height of the input raster (pixels,grid cells)
            im_width (float): width of the input raster (pixels,grid cells)

        Returns:
            2D NumPy: interpolated 2D latitude map 
        """        
        # Generate a 2D array of latitude    
        # for parallel computing
        if self.parallel:
            import dask.array as da

            # For lat_min, lat_max values given at pixel centers    
            if location:
                lat_vals=da.linspace(lat_max, lat_min,  im_height).astype('float32')    
                lat_map=da.broadcast_to(lat_vals[:,np.newaxis],(im_height,im_width),chunks=self.chunk2D)
                           
            # For lat_min, lat_max values given at exterior pixel edges    
            if ~location:
                lat_step=(lat_max-lat_min)/im_height    
                lat_vals = da.linspace(lat_max-lat_step/2, lat_min+lat_step/2,  im_height)    
                lat_map=da.broadcast_to(lat_vals[:,np.newaxis],(im_height,im_width),chunks=self.chunk2D) 

        # for serial computing
        else:     
            if location:
                lat_vals=np.linspace(lat_max, lat_min, im_height).astype('float32')    
                lat_map=np.broadcast_to(lat_vals[:,np.newaxis],(im_height,im_width))  

            if ~location:
                lat_step=(lat_max-lat_min)/im_height    
                lat_vals = np.linspace(lat_max-lat_step/2, lat_min+lat_step/2,  im_height)    
                lat_map=np.broadcast_to(lat_vals[:,np.newaxis],(im_height,im_width))   

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


    def smoothDailyTemp(self, day_start, day_end, mask, daily_T):    
        """create smoothed daily temperature curve using 5th degree spline 

        Args:
            day_start (integer): first day of the data in Julian day 
            day_end (integer): last day of the data in Julian day 
            mask (2D integer array): administrative mask of 0's and 1's, if user doesn't create this it comes in as all 1's
            daily_T (3D float array): daily temperature

        Returns:
            3D NumPy array: 5th degree spline smoothed temperature
        """        
        nlats=daily_T.shape[0]
        nlons=daily_T.shape[1] 
        ndays=daily_T.shape[2]

        if self.parallel:   
            import dask
            import dask.array as da

            # spline fitting, isolate the computation in a func so we can delay/parallelize
            def polyfit_polyval(x,y,deg):
                coefs=np.polynomial.polynomial.polyfit(x,y,deg=deg).astype('float32')
                spline=np.polynomial.polynomial.polyval(x,coefs).astype('float32')
                return spline.astype('float32') 

            # prepare func inputs
            deg=5  # polynomial degree for spline fitting
            days = np.arange(day_start,day_end+1).astype('int32') # x values    
            npoints=nlats*self.chunk3D[1] # total number of grid points
            
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                # replace any nan in the data with zero (nans may be present under a mask)   
                mask3D=da.broadcast_to(mask[:,:,np.newaxis],(nlats,nlons,ndays)).rechunk(chunks=self.chunk3D)
                data=da.where(mask3D==0,np.float32(0),np.float32(daily_T))     
                # make time the first dim, collapse lat and lon because polyfit and polyval only work on data up to 2 dimensions, chunk along space            
                data2D=data.transpose(2,0,1).reshape(ndays,-1).rechunk({0:-1,1:npoints}) 

            # delay data so it's only passed to computations once
            days=dask.delayed(days)
            delayed_chunks=data2D.to_delayed().ravel()
            
            # create list of delayed compute tasks
            task_list = [dask.delayed(polyfit_polyval)(days,dchunk,deg) for dchunk in delayed_chunks]
            
            # compute in parallel
            results_list=dask.compute(*task_list)  # dask compute/convert to numpy
            
            # concatenate resulting chunks
            interp_daily_temp=np.concatenate(results_list)
            del results_list

            # 2D back to 3D
            interp_daily_temp=interp_daily_temp.reshape(nlats,nlons,ndays)  

        else:
            ### FOR parallel=False COMPUTE WITHOUT DASK ###
            days = np.arange(day_start,day_end+1) # x values    
            # replace any nan (i.e. if there is a mask) in the data with zero    
            mask3D=np.broadcast_to(mask[:,:,np.newaxis],(nlats,nlons,ndays))
            data=np.where(mask3D==0,0,daily_T)     
            # make time first dim, collapse lat and lon because polyfit and polyval only work on data up to 2 dimensions
            data2D=data.transpose(2,0,1).reshape(days.shape[0],-1) # every column is a set of y values    
            del data    

            # do the spline fitting    
            quad_spl=np.polynomial.polynomial.polyfit(days,data2D,deg=5)    
            interp_daily_temp=np.polynomial.polynomial.polyval(days,quad_spl)    

            # reshape and reapply mask   
            interp_daily_temp=interp_daily_temp.reshape(mask3D.shape[0],mask3D.shape[1],-1).astype('float32')    
            interp_daily_temp=np.where(mask3D==0,np.nan,interp_daily_temp)

        return interp_daily_temp   

#----------------- End of file -------------------------#    