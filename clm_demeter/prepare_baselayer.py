import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # to generate plots
from netCDF4 import Dataset     # to work with NetCDF files
import os                       # operating system interface
from scipy.interpolate import griddata

__author__ = 'Min Chen and Eva Sinha'
__email__ = 'min.chen@pnnl.gov and eva.sinha@pnnl.gov'
__copyright__ = 'Copyright (c) 2017, Battelle Memorial Institute'

# ------------------------------------------------------------------------
def interpolate_coarse_fine(data, coarse_grid_coord, fine_grid_cooord):
    """ Interpolate values of variable from coarse grid to fine grid
    :param data:             subset(only land cells) of coarse grid coordinates along with values for variable
    :param coarse_grid_coord pandas dataframe containing latitude and longitude coordinate of coarse grid
    :param fine_grid_coord   pandas dataframe containing latitude and longitude coordinate of fine grid
    
    """
    # Join the subset grid coordinate data with full coordinate data
    coarse_grid_data = pd.merge(coarse_grid_coord, data, how='left')

    # Interpolate from coarse grid to finer resolution grid. 
    # Convert dataframe to ndarray for method='nearest' to work
    data_fine = griddata(points=coarse_grid_data[['Latcoord','Loncoord']].values, 
                         values=coarse_grid_data[data.columns[2]].values, 
                         xi=fine_grid_coord.values, 
                         method='nearest')
    
    return data_fine
# ------------------------------------------------------------------------

project_dir = '/pic/projects/im3/lulc_thrust_area/minchen/Demeter_0.05deg_trial/Demeter_0.05deg_trial'

# Base layer file for Plant Function Type (PFT)
# baselayer_pft_file = os.path.join(project_dir, 'mksrf_78pft_landuse_rc2000_c130927.nc')
baselayer_pft_file = 'mksrf_78pft_landuse_rc2000_c130927.nc'

# Open netCDF file
dataset = Dataset(baselayer_pft_file, mode='r') # file handle, open in read only mode

# Read latitude, longitude values
lat = dataset.variables['LAT']  # dim = lat
lon = dataset.variables['LON']  # dim = lon

lat_xy = dataset.variables['LATIXY'] # dim = (lat, lon)
lon_xy = dataset.variables['LONGXY'] # dim = (lat, lon)
landmask = dataset.variables['LANDMASK'] # dim = (lat, lon)

# Read data for PFT variable [pft (79), lat (3600), lon (7200)]
pft = dataset.variables['PCT_PFT']

close the file
dataset.close()

# flip the data in latitude so North Hemisphere is up on the plot
# flipud will convert the netcdf variable to numpy array
lat      = np.flipud(lat) 
lat_xy   = np.flipud(lat_xy)
lon_xy   = np.flipud(lon_xy)
landmask = np.flipud(landmask)
pft      = pft[:,::-1,:] 

# PFT values are converted from percentage (0-100) to fraction (0-1)
pft = pft*0.05*0.05*0.01; # unit conversion

# Generating land see mask
# pft_tot = np.sum(pft, axis=0) # Sum values for all PFTs
# landmask = (pft_tot > 0)      # identify land grid cells
landmask[(lat <= -55),:] = 0;   # assigning 0 to Antartica [lat less than -55] grid cells

# Reshape to convert lat lon 2D array values to 1 dimension
pft = np.reshape(pft, (79,-1))      # from [79, 3600, 7200] to [79, 3600x7200]
lat_xy = np.reshape(lat_xy, -1)     # from [3600, 7200] to [3600x7200, ]
lon_xy = np.reshape(lon_xy, -1)     # from [3600, 7200] to [3600x7200, ]
landmask = np.reshape(landmask, -1) # from [3600, 7200] to [3600x7200, ]

# Swap axes for pft array from [79, 3600x7200] to [3600x7200, 79]
pft = np.swapaxes(pft, 0, 1)

# Round lat and lon values to 3 decimal digits
lat_xy = lat_xy.round(3)
lon_xy = lon_xy.round(3)

# Generate FID for various grid cells (range from 0:3600x7200)
fid = np.array(range(np.shape(lat)[0] * np.shape(lon)[0]))

# Generate PFT column names
PFT_names = ['PFT'+ str(x) for x in range(1,pft.shape[1]+1)]
PFT_names = np.hstack(('Latcoord', 'Loncoord', PFT_names, 'FID'))

# Stack lat, lon, pft, and fid in a single array
pft = np.column_stack((lat_xy, lon_xy, pft, fid))

# Subset reshaped PFT values to only keep land grid cells
pft = pft[(landmask >0), ]

# Convert pft numpy array to pandas dataframe
pft = pd.DataFrame(pft, columns=PFT_names)

# Round lat and lon values to 3 decimal digits
pft.Latcoord = pft.Latcoord.round(3)
pft.Loncoord = pft.Loncoord.round(3)

# Read ESACCI land cover, nutrient availability, and soil quality files
esa = pd.read_csv('ESACCI-LC-aggregated-0.25Deg-merged1992-base.csv')
nv = pd.read_csv('000_nutrientavail_hswd_NA_0.25.csv', header=None)
sn = pd.read_csv('001_soilquality_hswd_0.25.csv', header=None)

# Modifying column names for nv and sn dataframes
nv.columns = ['FID', 'nv']
sn.columns = ['FID', 'sn']

# Join all three dataframes based on FID
lu_nv_sn = pd.merge(pd.merge(esa, nv, on='FID'), sn, on='FID')

# Fine grid coordinates
fine_grid_coord = pd.DataFrame({'Latcoord':lat_xy, 'Loncoord':lon_xy}).round(3)

# Coarse grid coordinates
[lat_coarse, lon_coarse] = np.meshgrid(np.arange(-89.875,89.875,0.25), np.arange(-179.875,179.875,0.25)) # 0.25 degree meshgrid
lat_xy_coarse = np.reshape(lat_coarse, -1)     # from [1439, 719] to [1439x719, ]
lon_xy_coarse = np.reshape(lon_coarse, -1)     # from [1439, 719] to [1439x719, ]
coarse_grid_coord = pd.DataFrame({'Latcoord':lat_xy_coarse, 'Loncoord':lon_xy_coarse})

# Interpolate values of variable from coarse grid to fine grid using nearest neigbor
nv_fine = interpolate_coarse_fine(lu_nv_sn[['Latcoord','Loncoord','nv']], coarse_grid_coord, fine_grid_coord)
sn_fine = interpolate_coarse_fine(lu_nv_sn[['Latcoord','Loncoord','sn']], coarse_grid_coord, fine_grid_coord)
aez_id_fine = interpolate_coarse_fine(lu_nv_sn[['Latcoord','Loncoord','aez_id']], coarse_grid_coord, fine_grid_coord)
region_id_fine = interpolate_coarse_fine(lu_nv_sn[['Latcoord','Loncoord','region_id']], coarse_grid_coord, fine_grid_coord)

# Create dataframe containing FID, lon, lat, nut. avail., soil quality, regionAEZ,regionID, AEZID
reg_aez_fine = region_id_fine*100 + aez_id_fine
lu_nv_sn_fine_grid = pd.DataFrame({'FID':fid, 'lat':lat_xy, 'lon':lon_xy, 'nv':nv_fine, 'sn':sn_fine,
                                  'regAEZ':reg_aez_fine,'region_id':region_id_fine, 'aez_id':aez_id_fine})

# Remove NaN rows
lu_nv_sn_final = lu_nv_sn_fine_grid[~np.isnan(lu_nv_sn_fine_grid.sum(axis=1, skipna=False))]

# Join dataframes for PFT values and final dataframe for land use, nutrient availability, and soil quality
pft_lu_nv_sn = pd.merge(pft, lu_nv_sn_final, on='FID', how='left')

# Add OBJECTID
pft_lu_nv_sn = pft_lu_nv_sn.assign(OBJECTID = np.arange(1, pft_lu_nv_sn.shape[0]+1, 1))

# The lines below for changing type to integer do not work since there are NaN values in few rows
# Change data type for FID, regAEZ, region_id, and aez_id
#pft_lu_nv_sn.FID = pft_lu_nv_sn.FID.astype('int64')
#pft_lu_nv_sn.regAEZ = pft_lu_nv_sn.regAEZ.astype('int64')
#pft_lu_nv_sn.region_id = pft_lu_nv_sn.region_id.astype('int64')
#pft_lu_nv_sn.aez_id = pft_lu_nv_sn.aez_id.astype('int64')

# Write data in csv files
nv.df = pft_lu_nv_sn[['FID','nv']]                          # Subset nutrient availability data
nv.df.to_csv(path_or_buf ='nv_constraint.csv', index=False) # Write nurtient availability data 
   
sn.df = pft_lu_nv_sn[['FID','sn']]                          # Subset soil quality data
sn.df.to_csv(path_or_buf ='sn_constraint.csv', index=False) # Write soil quality data 

pft.df = pft_lu_nv_sn.drop(['lat','lon','nv','sn'], axis=1)  # Subset PFT data
pft.df.to_csv(path_or_buf ='baselayerdata.csv', index=False) # Write PFT data 