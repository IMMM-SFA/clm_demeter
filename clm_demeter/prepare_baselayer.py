import pandas as pd
import numpy as np
from netCDF4 import Dataset     # to work with NetCDF files
from scipy.interpolate import griddata

__author__ = 'Min Chen and Eva Sinha'
__email__ = 'min.chen@pnnl.gov and eva.sinha@pnnl.gov'
__copyright__ = 'Copyright (c) 2017, Battelle Memorial Institute'

class PrepareBaselayer:
    """Generate CLM base layer for plant fuctional types
    :param orig_baselayer_netcdf_file:   Base layer file in netcdf format for Plant Function Type (PFT)
    
    :param esacci_lu_file:               ESACCI land cover file
    
    :param nut_avail_file:               Nutrient availability file
    
    :param soil_qual_file:               Soil quality file
    
    :method read_baselayer                           Read netcdf file and create pft, lat, lon, and FID to numpy array
    
    :method convert_baselayer_to_dataframe           Convert baselayer to dataframe format
    
    :method read_lu_nutavail_soilqual                Read ESACCI landuse, nutrient availability, 
                                                     and soil quality file
                                                     
    :method interpolate_coarse_fine                  Interpolate values of variable from coarse grid to fine grid

    
    :method convert_lu_nutavail_soilqual_fine_grid   Interpolate landuse, nutrient availability, 
                                                     and soil quality data from coarse grid to fine grid
    
    :method merge_pft_lu_nv_sn                       Join dataframes for PFT values and final dataframe 
                                                     for land use, nutrient availability, and soil quality
                                                     Save output to new files
    """
    
    def __init__(self, orig_baselayer_netcdf_file, esacci_lu_file, nut_avail_file, soil_qual_file):
        
        self.orig_baselayer_netcdf_file = orig_baselayer_netcdf_file
        self.esacci_lu_file = esacci_lu_file
        self.nut_avail_file = nut_avail_file
        self.soil_qual_file = soil_qual_file
        
        # Read netcdf file and create pft, lat, lon, and FID to numpy array
        self.pft, self.lat_xy, self.lon_xy, self.fid = self.read_baselayer(lat_var='LAT', 
                                                                           lon_var='LON', 
                                                                           lat_xy_var='LATIXY', 
                                                                           lon_xy_var='LONGXY',
                                                                           landmask_var='LANDMASK', 
                                                                           pft_var='PCT_PFT')
        
        # Convert baselayer to dataframe format
        self.pft_df = self.convert_baselayer_to_dataframe(latitude="latitude", longitude="longitude")

        # Read ESACCI landuse, nutrient availability, and soil quality file
        self.lu_nv_sn = self.read_lu_nutavail_soilqual()

        # Interpolate landuse, nutrient availability, and soil quality data 
        # from coarse grid to fine grid
        self.lu_nv_sn_final = self.convert_lu_nutavail_soilqual_fine_grid(latitude="Latcoord", longitude="Loncoord")
        
        # Join dataframes for PFT values and final dataframe for land use, nutrient availability, and soil quality
        # Save output to new files
        self.merge_pft_lu_nv_sn(join_variable='FID')
        
    def read_baselayer(self, lat_var='LAT', lon_var='LON', lat_xy_var='LATIXY', lon_xy_var='LONGXY',
                       landmask_var='LANDMASK', pft_var='PCT_PFT'):
        """ Read netcdf file and create pft, lat, lon, and FID to numpy array
        :param lat_var:         variable name for latitude
        
        :param lon_var:         variable name for longitude
        
        :param lat_xy_var:      variable name for latitude grid
        
        :param lon_xy_var:      variable name for longitude grid
        
        :param lankdmask_var:   variable name for landmask
        
        :param pft_var:         variable name for pft
        
        :return                  return numpy array for pft, latitude, longitude, and FID 
        """

        # Open netCDF file
        dataset = Dataset(self.orig_baselayer_netcdf_file, mode='r') # file handle, open in read only mode

        # Read latitude, longitude values
        lat = dataset.variables[lat_var]  # dim = lat
        lon = dataset.variables[lon_var]  # dim = lon

        lat_xy = dataset.variables[lat_xy_var] # dim = (lat, lon)
        lon_xy = dataset.variables[lon_xy_var] # dim = (lat, lon)
        landmask = dataset.variables[landmask_var] # dim = (lat, lon)

        # Read data for PFT variable [pft (79), lat (3600), lon (7200)]
        pft = dataset.variables[pft_var]

        # flip the data in latitude so North Hemisphere is up on the plot
        # flipud will convert the netcdf variable to numpy array
        lat      = np.flipud(lat) 
        lat_xy   = np.flipud(lat_xy)
        lon_xy   = np.flipud(lon_xy)
        landmask = np.flipud(landmask)
        pft      = pft[:, ::-1, :] 

        # PFT values are converted from percentage (0-100) to fraction (0-1)
        delta_lat = round(abs(lat[1] - lat[0]), 3)
        delta_lon = round(abs(lon[1] - lon[0]), 3)
        per_to_frac = 0.01     # for converting from percentage (0-100) to fraction (0-1)
        
        pft = pft * delta_lat * delta_lon * per_to_frac; # unit conversion

        # Generating land see mask
        landmask[(lat <= -55), :] = 0;   # assigning 0 to Antartica [lat less than -55] grid cells

        # Reshape to convert lat lon 2D array values to 1 dimension
        pft = np.reshape(pft, (pft.shape[0], -1)) # from [79, 3600, 7200] to [79, 3600x7200]
        lat_xy = np.reshape(lat_xy, -1)           # from [3600, 7200] to [3600x7200, ]
        lon_xy = np.reshape(lon_xy, -1)           # from [3600, 7200] to [3600x7200, ]
        landmask = np.reshape(landmask, -1)       # from [3600, 7200] to [3600x7200, ]

        # Swap axes for pft array from [79, 3600x7200] to [3600x7200, 79]
        pft = np.swapaxes(pft, 0, 1)

        # Round lat and lon values to 3 decimal digits
        lat_xy = lat_xy.round(3)
        lon_xy = lon_xy.round(3)

        # Generate FID for various grid cells (range from 0:3600x7200)
        fid = np.array(range(np.shape(lat)[0] * np.shape(lon)[0]))

        # Stack lat, lon, pft, and fid in a single array
        pft = np.column_stack((lat_xy, lon_xy, pft, fid))

        # Subset reshaped PFT values to only keep land grid cells
        pft = pft[(landmask > 0), ]

        # close the file
        dataset.close()
        
        return (pft, lat_xy, lon_xy, fid)
    
    def convert_baselayer_to_dataframe(self, latitude="latitude", longitude="longitude"):
        """ Convert baselayer to dataframe format
        :param latitude:         variable for saving latitude values
        
        :param longitude:        variable for saving longitude values
        
        :return                  return baselayer converted to dataframe containing 
                                 latitude, longitude, plant functional type, and FID
        """
        
        # Generate PFT column names
        PFT_names = ['PFT'+ str(x) for x in range(1, self.pft.shape[1] + 1 - len((latitude, longitude, 'FID')))]
        PFT_names = np.hstack((latitude, longitude, PFT_names, 'FID'))
        
        # Convert pft numpy array to pandas dataframe
        pft_df = pd.DataFrame(self.pft, columns=PFT_names)

        # Round lat and lon values to 3 decimal digits
        pft_df[latitude] = pft_df[latitude].round(3)
        pft_df[longitude] = pft_df[longitude].round(3)
        
        return (pft_df)


    def read_lu_nutavail_soilqual(self):
        """ Read ESACCI landuse, nutrient availability, and soil quality file
                                 and merge the three files into a single dataframe
        :return                  return merged landuse, nutrient availability and soil quality dataframe
        """
            
        # Read ESACCI land cover, nutrient availability, and soil quality files
        esa = pd.read_csv(self.esacci_lu_file)
        nv = pd.read_csv(self.nut_avail_file, header=None)
        sn = pd.read_csv(self.soil_qual_file, header=None)

        # Modifying column names for nv and sn dataframes
        nv.columns = ['FID', 'nv']
        sn.columns = ['FID', 'sn']

        # Join all three dataframes based on FID
        lu_nv_sn = pd.merge(pd.merge(esa, nv, on='FID'), sn, on='FID')
        
        return lu_nv_sn

    def interpolate_coarse_fine(self, coarse_grid_coord, fine_grid_coord, variable, latitude="latitude", longitude="longitude"):
        """ Interpolate values of variable from coarse grid to fine grid
        :param data:             subset(only land cells) of coarse grid coordinates along with values for variable
        
        :param coarse_grid_coord pandas dataframe containing latitude and longitude coordinate of coarse grid
        
        :param fine_grid_coord   pandas dataframe containing latitude and longitude coordinate of fine grid
        
        :param variable:         variable for interpolation to fine grid
        
        :param latitude:         variable for saving latitude values
        
        :param longitude:        variable for saving longitude values
        
        :return                  value of the variable interpolated to the fine grid resolution
        """
        # Join the subset grid coordinate data with full coordinate data
        coarse_grid_data = pd.merge(coarse_grid_coord, self.lu_nv_sn, how='left')
    
        # Interpolate from coarse grid to finer resolution grid.
        # Convert dataframe to ndarray for method='nearest' to work
        data_fine = griddata(points=coarse_grid_data[[latitude, longitude]].values,
                             values=coarse_grid_data[variable].values,
                             xi=fine_grid_coord.values,
                             method='nearest')
        return data_fine
                         
    def convert_lu_nutavail_soilqual_fine_grid(self, latitude="Latcoord", longitude="Loncoord"):
        """ Interpolate landuse, nutrient availability, and soil quality data
            from coarse grid to fine grid
        :param latitude:         variable for saving latitude values
        
        :param longitude:        variable for saving longitude values
            
        :return                  return merged landuse, nutrient availability, and soil quality
                                 at a fine grid resolution
        """
        # Fine grid coordinates
        fine_grid_coord = pd.DataFrame({latitude:self.lat_xy, longitude:self.lon_xy}).round(3)

        # Coarse grid coordinates (0.25 deg mesh grid)
        coarse_min_lat = -89.875
        coarse_max_lat = 89.875
        coarse_min_lon = -179.875
        coarse_max_lon = 179.875
        step = 0.25
        
        [lat_coarse, lon_coarse] = np.meshgrid(np.arange(coarse_min_lat, coarse_max_lat, step), 
                                               np.arange(coarse_min_lon, coarse_max_lon, step))
        
        lat_xy_coarse = np.reshape(lat_coarse, -1)     # from [1439, 719] to [1439x719, ]
        lon_xy_coarse = np.reshape(lon_coarse, -1)     # from [1439, 719] to [1439x719, ]
        coarse_grid_coord = pd.DataFrame({latitude:lat_xy_coarse, longitude:lon_xy_coarse})

        # Interpolate values of variable from coarse grid to fine grid using nearest neigbor
        nv_fine = self.interpolate_coarse_fine(coarse_grid_coord, fine_grid_coord, 'nv', latitude, longitude)
        sn_fine = self.interpolate_coarse_fine(coarse_grid_coord, fine_grid_coord, 'sn', latitude, longitude)
        aez_id_fine = self.interpolate_coarse_fine(coarse_grid_coord, fine_grid_coord, 'aez_id', latitude, longitude)
        region_id_fine = self.interpolate_coarse_fine(coarse_grid_coord, fine_grid_coord, 'region_id', latitude, longitude)

        # Create dataframe containing FID, lon, lat, nut. avail., soil quality, regionAEZ,regionID, AEZID
        reg_aez_fine = region_id_fine*100 + aez_id_fine
        lu_nv_sn_fine_grid = pd.DataFrame({'FID':self.fid, 'lat':self.lat_xy, 'lon':self.lon_xy, 'nv':nv_fine, 'sn':sn_fine,
                                           'regAEZ':reg_aez_fine, 'region_id':region_id_fine, 'aez_id':aez_id_fine})

        # Remove NaN rows
        lu_nv_sn_final = lu_nv_sn_fine_grid[~np.isnan(lu_nv_sn_fine_grid.sum(axis=1, skipna=False))]
        
        return lu_nv_sn_final

    def merge_pft_lu_nv_sn(self, join_variable='FID'):
        """ Join dataframes for PFT values and final dataframe for land use, 
            nutrient availability, and soil quality
            Save output to new files 
        :param latitude:         variable to perform the join on 
        """
        # Join dataframes for PFT values and final dataframe for land use, nutrient availability, and soil quality
        pft_lu_nv_sn = pd.merge(self.pft_df, self.lu_nv_sn_final, on=join_variable, how='left')

        # Add OBJECTID
        pft_lu_nv_sn = pft_lu_nv_sn.assign(OBJECTID = np.arange(1, pft_lu_nv_sn.shape[0] + 1, 1))

        # Write data in csv files
        nv_df = pft_lu_nv_sn[['FID', 'nv']]                             # Subset nutrient availability data
        nv_df.to_csv(path_or_buf ='nv_constraint.csv', index=False)     # Write nurtient availability data 
   
        sn_df = pft_lu_nv_sn[['FID', 'sn']]                             # Subset soil quality data
        sn_df.to_csv(path_or_buf ='sn_constraint.csv', index=False)     # Write soil quality data 

        pft_df = pft_lu_nv_sn.drop(['lat', 'lon', 'nv', 'sn'], axis=1)  # Subset PFT data
        pft_df.to_csv(path_or_buf ='baselayerdata.csv', index=False)    # Write PFT data 
