"""
PyAEZ: Additional calculations used throughout AEZ modules
2020: N. Lakmal Deshapriya
2022: Swun Wunna Htet, K. Boonma
"""

import numpy as np
import xarray as xr
# from scipy.interpolate import interp1d
# import pandas as pd
import rioxarray as rio
# try:
#     import gdal
# except:
#     from osgeo import gdal



class UtilitiesCalc(object):

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

        # we've assigned this info to the data time dim already so we don't need this here
        # doy_middle_of_month = np.arange(0,12)*30 + 15 # Calculate doy of middle of month

        # create a vector of daily julian day for a year
        time_daily=np.arange(cycle_begin,cycle_end+1)

        # locate grids that are all nan
        # this assumes no intermittent nans (each grid cell has either 12 finite or all nan)
        # can modify this in the future to look for nan anywhere, for now using time zero
        nanmask=xr.where(monthly_vector[0,:,:].isnull(),0,1) 

        # this comp originally executed on a vector of 12 values (time dim) from a single grid cell
        # instead, we implement xarray below which works on the entire spatial domain with a single line of code
        # inter_fun = interp1d(doy_middle_of_month, monthly_vector, kind='quadratic', fill_value='extrapolate')
        # daily_vector = inter_fun( np.arange(cycle_begin,cycle_end+1) )

        # prep monthly data array
        # first find grids that are all nan and change to zeros, else scipy interp outputs all nan for everything with no warning error
        monthly_vector=monthly_vector.where(nanmask==1,0)       
        # second collapse lat/lon bcz 'quadratic' option is only availabe on interp1d (not interpn) which can only accept 2D array
        monthly_vector=monthly_vector.stack(space=('y','x')) # (time,y,x)-->(time,space)
        
        # vectorized comp eliminates 'for' loops in ClimateRegime.setMonthlyClimateData
        # xarray interp is using scipy interp1d here
        daily_vector=monthly_vector.interp(time=time_daily,method="quadratic",kwargs={"fill_value":"extrapolate"}) 
        
        # reshape and add back nan cells
        daily_vector=daily_vector.unstack('space')
        daily_vector=daily_vector.where(nanmask==1)

        if no_minus_values:
            # daily_vector[daily_vector<0] = 0
            daily_vector=xr.where(daily_vector<0,0,daily_vector)

        # should probably implement some dimension checking here
        # since we're no longer pre-allocating arrays in ClimateRegime.setMonthlyClimateData

        return daily_vector

    def averageDailyToMonthly(self, daily_vector):
        """Aggregating daily data into monthly data

        Args:
            daily_vector (1D NumPy): daily data array

        Returns:
            1D NumPy: Monthly data array
        """       

        monthly_vector = np.zeros((daily_vector.shape[0],daily_vector.shape[1],12))

        monthly_vector[:,:,0] = daily_vector[:,:,:31].sum(axis=2)/31
        monthly_vector[:,:,1] = daily_vector[:,:,31:59].sum(axis=2)/28
        monthly_vector[:,:,2] = daily_vector[:,:,59:90].sum(axis=2)/31
        monthly_vector[:,:,3] = daily_vector[:,:,90:120].sum(axis=2)/30
        monthly_vector[:,:,4] = daily_vector[:,:,120:151].sum(axis=2)/31
        monthly_vector[:,:,5] = daily_vector[:,:,151:181].sum(axis=2)/30
        monthly_vector[:,:,6] = daily_vector[:,:,181:212].sum(axis=2)/31
        monthly_vector[:,:,7] = daily_vector[:,:,212:243].sum(axis=2)/31
        monthly_vector[:,:,8] = daily_vector[:,:,243:273].sum(axis=2)/30
        monthly_vector[:,:,9] = daily_vector[:,:,273:304].sum(axis=2)/31
        monthly_vector[:,:,10] = daily_vector[:,:,304:334].sum(axis=2)/30
        monthly_vector[:,:,11] = daily_vector[:,:,334:].sum(axis=2)/31        

        return monthly_vector

    def generateLatitudeMap(self, lats, nlons):
    # def generateLatitudeMap(self, lat_min, lat_max, im_height, im_width):
        """Create latitude map from input geographical extents

        Args:
            lat_min (float): the minimum latitude
            lat_max (float): the maximum latitude
            im_height (float): height of the input raster (pixels,grid cells)
            im_width (float): width of the input raster (pixels,grid cells)

        Returns:
            2D NumPy: interpolated 2D latitude map 
        """        
        # lat_lim = np.linspace(lat_min, lat_max, im_height)
        # lon_lim = np.linspace(1, 1, im_width) # just temporary lon values, will not affect output of this function.
        # [X_map,Y_map] = np.meshgrid(lon_lim,lat_lim)
        # lat_map = np.flipud(Y_map)

        # precision issue arise from above, why not just take in all lats as an input instead of recomputing them with linspace
        lat_map=np.tile(lats[:,np.newaxis],(1,nlons))

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

        # # Read random image to get projection data
        # img = gdal.Open(ref_raster_path)
        # # allocating space in hard drive
        # driver = gdal.GetDriverByName("GTiff")
        # outdata = driver.Create(out_path, img.RasterXSize, img.RasterYSize, 1, gdal.GDT_Float32)
        # # set image paramenters (imfrormation related to cordinates)
        # outdata.SetGeoTransform(img.GetGeoTransform())
        # outdata.SetProjection(img.GetProjection())
        # # write numpy matrix as new band and set no data value for the band
        # outdata.GetRasterBand(1).WriteArray(numpy_raster)
        # outdata.GetRasterBand(1).SetNoDataValue(-999)
        # # flush data from memory to hard drive
        # outdata.FlushCache()
        # outdata=None

        # go back to above eventually, currently it is erroring
        if 'if' in ref_raster_path[-4:]:
            # get spatial reference from tif
            spatial_ref=rio.open_rasterio(ref_raster_path)['spatial_ref']
            mask=rio.open_rasterio(ref_raster_path).squeeze()        
        elif 'nc' in ref_raster_path[-3:]:
            # get spatial ref from netcdf
            spatial_ref=xr.open_dataset(ref_raster_path)['spatial_ref']
            vname=ref_raster_path.split('/')[-1].split('_')[0]
            try:
                mask=xr.open_dataset(ref_raster_path)[vname].data
            except:
                print('no variable named mask found in netcdf file. rewrite netcdf with variable named mask or use a tif mask')
        else:
            # throw error
            print('ERROR: '+ref_raster_path+' is not a tif or nc file. Input mask from georeferenced tif or nc file')
        
        xr_data=xr.DataArray(numpy_raster,dims=['y','x'])
        xr_data=xr_data.assign_coords({'spatial_ref':spatial_ref})
        xr_data=xr_data.where(mask==1).fillna(-999).astype('int16')
        
        try:
            xr_data.rio.to_raster(out_path)
        except:
            print('ERROR: data not written to tif')

        # may want to look into how to add more metadata to the tif output
        # idk what is standard for geotif
        

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
