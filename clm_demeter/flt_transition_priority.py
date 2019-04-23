import pandas as pd
import numpy as np
import numpy.ma as ma           # masked array module
from netCDF4 import Dataset     # to work with NetCDF files
import os                       # operating system interface

__author__ = 'Min Chen and Eva Sinha'
__email__  = 'min.chen@pnnl.gov and eva.sinha@pnnl.gov'
__copyright__ = 'Copyright (c) 2017, Battelle Memorial Institute'

class FLTTransitionPriorities:
    """Develop transition priorities for Final Land Types (FLTs)
    :param: esacci_to_flt_file        File name storing mapping between ESACCI land classes and FLTs
    
    :param sta_yr                     Start year for ESACCI data
    
    :param end_yr                     End year for ESACCI data
    
    :param: fname_sta:                file part of the filename
        
    :param: fname_end:                second part of the filename
        
    :param: lu_var:                   land class variable to read from the netCDF file
       
    :method ESACCI_to_FLT_map:        Create a dictionary of ESACCI land classes to FLTs

    :method estimate_FLT_transitions: Read ESACCI land class file for various years 
                                      and develop FLTs transitions
    
    :method read_lu_file:             Read ESACCI land class file in netCDF format
    
    :method convert_lu_FLT:           Convert ESACCI land use classes to FLTs
    
    :method count_transition_cells:   Count number of grid cells where land class changes from one value to another
    
    :method estimate_trans_priorities: Convert transitions into a table for transition priorities
    """
    
    def __init__(self, esacci_to_flt_file, sta_yr, end_yr, fname_sta, fname_end, lu_var):
        
        self.esacci_to_flt_file = esacci_to_flt_file
        self.sta_yr = sta_yr
        self.end_yr = end_yr
        self.fname_sta = fname_sta
        self.fname_end = fname_end
        self.lu_var = lu_var
        
        # Create a dictionary of ESACCI land classes to Final Land Types (FLTs)
        self.d, self.n_flt = self.ESACCI_to_FLT_map()
        
        # Read ESACCI land class file for various years and develop FLTs transitions
        self.transitions = self.estimate_FLT_transitions()
        
        # Calculate transition priorities
        self.trans_prior = self.estimate_trans_priorities()
    
    def ESACCI_to_FLT_map(self):
        """Create a dictionary of ESACCI land classes to FLTs
        :return:              Dictionary of {ESA_code: PFT_code} and number of unique FLTs
        """
        #  create a dictionary of {ESA_code: PFT_code}
        d = {}
        with open(self.esacci_to_flt_file) as file:
            
            # Skip the column names
            file.readline()
    
            for i, line in enumerate(file):
                s = line.strip().split('\t')
                d[s[1]] = s[2]  # Second column (1) is the key and third column (2) is the value
    
        # Count number of unique FLTs in the dictionary
        n_flt = len(set(d.values()))
    
        return (d, n_flt)
    
    def estimate_FLT_transitions(self):
        """Read ESACCI land class file for various years and develop FLTs transitions
        :return:              numpy array of FLTs transitions
        """
    
        # Empty array for storing transitions
        transitions = np.zeros((self.n_flt, self.n_flt), dtype=int)
    
        for year in np.arange(self.sta_yr, self.end_yr):
            
            # Read ESACCI land class file in netCDF format for year and year+1
            lccs_t1 = self.read_lu_file(year)
            lccs_t2 = self.read_lu_file(year + 1)
        
            # Convert ESACCI land use classes to Final Land Types (FLTs)
            lccs_t1 = self.convert_lu_FLT(lccs_t1)
            lccs_t2 = self.convert_lu_FLT(lccs_t2)
        
            # Use list comprehension to estimate transitions from lccs_t1 to lccs_t2 for all combination of land classes
            transition_list = [self.count_transition_cells(lccs_t1, lccs_t2, i, j) for i in np.arange(self.n_flt) for j in np.arange(self.n_flt)]

            # Reshape transitions into an array
            transitions = transitions + np.reshape(transition_list, (self.n_flt, self.n_flt))
        
        return (transitions)
    
    def read_lu_file(self, year):
        """Read ESACCI land class file in netCDF format        
        :param: year:           year for the ESACCI land class file 
        
        :return:                ESACCI land use value in numpy array format
        """
        
        # Filename for ESACCI land use file in netcdf format
        nc_file = self.fname_sta + str(year) + self.fname_end
        
        # Open netCDF file
        dataset = Dataset(nc_file, mode='r') # file handle, open in read only mode

        # Switch off automatic conversion to masked arrays and variable scaling
        # Without this switch off the negative values will be read incorrectly
        dataset.variables[self.lu_var].set_auto_maskandscale(False)

        # Read data for lccs_class variable [lat (64800), lon (129600)]
        # Land cover class defined in LCCS
        lccs = dataset.variables[self.lu_var][:] # Read variable and convert to numpy array
        
        return (lccs)
    
    def convert_lu_FLT(self, lccs):
        """Convert ESACCI land use classes to FLTs
        :param: lccs          ESACCI land use values in numpy array format
    
        :return:              land use classes converted to FLTs
        """

        # for each ESACCI land class
        for lc_lccs in self.d.keys():
            
            # Replace ESACCI land class with the FLTs
            lccs[lccs == int(lc_lccs)] = int(self.d[lc_lccs])
            
        return (lccs)
    
    def count_transition_cells(self, lccs_t1, lccs_t2, val_1, val_2):
        """Count number of grid cells where land class changes from val_1 to val_2
        :param: lccs_t1:      ESACCI land class array at time = T
        
        :param: lccs_t2:      ESACCI land class array at time = T + 1
        
        :param: val_1:        land class value to identify in lccs_1 
        
        :param: val_2:        land class value to identify in lccs_2
        
        :return:              Number of grid cells where land class changes from val_1 to val_2
        """

        # Mask grid cells that are not equal to a given value
        ma_val_t1 = ma.masked_not_equal(lccs_t1, val_1)
        ma_val_t2 = ma.masked_not_equal(lccs_t2, val_2)      
        
        # Replace unmasked values with 99
        ma_val_t1[~ma_val_t1.mask] = 99
        ma_val_t2[~ma_val_t2.mask] = 99
        
        # Compare the two masked arrays to find locations where both arrays have unmasked values
        ma_equal = np.equal(ma_val_t1, ma_val_t2)
        
        # Replace masked values with 0
        # This step ensures that if ma_equal array has no unmasked cells then zero is returned
        ma_equal[ma_equal.mask] = 0
        
        # Return total number of grid cells where ma_val_t1 and ma_val_t2 are unmasked
        return(np.sum(ma_equal))
    
    def estimate_trans_priorities(self):
        """Convert transitions into a table for transition priorities
    
        :return      array of transition priorities
        """
    
        # Remove first row and first column
        tmp = self.transitions[1:self.n_flt, 1:self.n_flt]

        # Assign zero (highest priority) to diagonal element
        np.fill_diagonal(tmp, 0)

        # Rank items in each row and save in an array
        trans_prior = tmp.argsort(axis=1).argsort(axis=1)
        
        # Convert numpy array to pandas dataframe for writing
        flt_names = ['forest', 'shrub', 'grass', 'crops', 'urban', 'snow', 'sparse']
        df = pd.DataFrame(trans_prior, index=flt_names, columns=flt_names)

        # Write data in csv files
        df.to_csv('transition_priority.csv', index=True, header=True, sep=',')

        return (trans_prior)
