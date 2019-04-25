import pandas as pd
import numpy as np

__author__ = 'Min Chen and Eva Sinha'
__email__  = 'min.chen@pnnl.gov and eva.sinha@pnnl.gov'
__copyright__ = 'Copyright (c) 2017, Battelle Memorial Institute'

class CLMTransitionPriorities:
    """Develop transition priorities for CLM types
    :param: geo_trans_prior_file        File name storing transition priorioties for geographical factors
    
    :param: FLT_trans_prior_file        File name storing transition priorioties for FLTs
    
    :param: leaf_trans_prior_file       File name storing transition priorioties for leaf shape factors
    
    :param: pheno_trans_prior_file      File name storing transition priorioties for phenology factors
    
    :param: crop_trans_prior_file       File name storing transition priorioties for croptype factors
    
    :param: water_trans_prior_file      File name storing transition priorioties for water factors
    
    :method read_trans_prior_file:     Read transition priority file
    
    """
    
    def __init__(self, geo_trans_prior_file, FLT_trans_prior_file, leaf_trans_prior_file,
                 pheno_trans_prior_file, crop_trans_prior_file, water_trans_prior_file,
                 clm_type_factors_file):
        
        self.geo_trans_prior_file   = geo_trans_prior_file
        self.FLT_trans_prior_file   = FLT_trans_prior_file
        self.leaf_trans_prior_file  = leaf_trans_prior_file
        self.pheno_trans_prior_file = pheno_trans_prior_file
        self.crop_trans_prior_file  = crop_trans_prior_file
        self.water_trans_prior_file = water_trans_prior_file
        self.clm_type_factors_file  = clm_type_factors_file
        
        # Read various transition priorities
        self.geo_trans_prior   = self.read_trans_prior_file(geo_trans_prior_file)
        self.FLT_trans_prior   = self.read_trans_prior_file(FLT_trans_prior_file)
        self.leaf_trans_prior  = self.read_trans_prior_file(leaf_trans_prior_file)
        self.pheno_trans_prior = self.read_trans_prior_file(pheno_trans_prior_file)
        self.crop_trans_prior  = self.read_trans_prior_file(crop_trans_prior_file)
        self.water_trans_prior = self.read_trans_prior_file(water_trans_prior_file)
        self.clm_type_factors  = self.read_trans_prior_file(clm_type_factors_file)
        
        # Estimate CLM transition priroties by accounting for priorities for various factors
        self.clm_trans_prio = self.estimate_clm_transition_priorities()
        
    @staticmethod
    def read_trans_prior_file(trans_prior_file):
        """Read transition priority file for particular factor
        
        :return: DataFrame with column and row lables
        """
        
        factor_trans_prior = pd.read_csv(trans_prior_file, index_col=0, sep=',', keep_default_na=False)
        
        return (factor_trans_prior)
    
    def transition_priority_lookup(self, trans_prior_df, colname, row_index):
        """Lookup transition priority for the specified factor for the particular CLM type 
        
        :return: array with transition priorities for the specified factor for the particular CLM type
        """
        
        # Row length or number of CLM types
        nrow = self.clm_type_factors.shape[0]
        
        # Identify column values to lookup
        col_labels = np.array(self.clm_type_factors[colname]) 
        
        # Identify row values to lookup. The selected CLM type is repeated for all columns
        row_labels = np.repeat(col_labels[row_index], nrow)
        
        # Lookup transition priority for the specified row and column values
        trans_prior = trans_prior_df.lookup(row_labels, col_labels)
        
        return (trans_prior)

    def estimate_clm_transition_priorities(self):
        """Estimate CLM transition priroties by accounting for priorities for various factors
        
        :return: DataFrame with transition priorities among the various CLM types
        """
        
        # Row length or number of CLM types
        nrow = self.clm_type_factors.shape[0]
        
        # Empty dataframe for storing CLM transition priorities
        clm_trans_prio = pd.DataFrame(data=None, columns=self.clm_type_factors.index)
        
        for i in np.arange(nrow):

            # Empty dataframe
            trans_prior_tmp = pd.DataFrame(data=None, columns=self.clm_type_factors.columns, index=self.clm_type_factors.index)

            # Lookup transition priority for the specified factor for the particular CLM type
            trans_prior_tmp['Factor_Geo']      = self.transition_priority_lookup(self.geo_trans_prior,   'Factor_Geo', i)
            trans_prior_tmp['Factor_Type']     = self.transition_priority_lookup(self.FLT_trans_prior,   'Factor_Type', i)
            trans_prior_tmp['Factor_Leaf']     = self.transition_priority_lookup(self.leaf_trans_prior,  'Factor_Leaf', i)
            trans_prior_tmp['Factor_Pheno']    = self.transition_priority_lookup(self.pheno_trans_prior, 'Factor_Pheno', i)
            trans_prior_tmp['Factor_CropType'] = self.transition_priority_lookup(self.crop_trans_prior,  'Factor_CropType', i)
            trans_prior_tmp['Factor_Water']    = self.transition_priority_lookup(self.water_trans_prior, 'Factor_Water', i)
        
            # Sort DataFrame by multiple columns
            trans_prior_sorted = trans_prior_tmp.sort_values(by=['Factor_Geo', 'Factor_Type', 'Factor_Leaf', 
                                                                 'Factor_Pheno', 'Factor_CropType', 'Factor_Water'])

            # Create tmp dataframe with single row for the particular CLM type
            # and containing transition priority from this CLM type to all CLM types
            ordered_clm = pd.DataFrame(data=[np.arange(nrow)], columns=trans_prior_sorted.index, index=[trans_prior_tmp.index[i]])

            # Append the row of transition priority
            clm_trans_prio = clm_trans_prio.append(ordered_clm, sort=False)
        
        # Write data in csv files
        clm_trans_prio.to_csv('clm_transition_priority.csv', index=True, header=True, sep=',')
        
        return (clm_trans_prio)
