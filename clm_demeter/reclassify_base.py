import numpy as np
import pandas as pd


__author__ = 'Chris R. Vernon'
__email__ = 'chris.vernon@pnnl.gov'
__copyright__ = 'Copyright (c) 2017, Battelle Memorial Institute'
__license__ = 'BSD 2-Clause'


class ReclassBaselayer:
    """Reclassify CLM PFTs in the baselayer for Demeter to Demeter final landcover classes.

    :param clm_baselayer_file:              Full path with file name and extension to
                                            the CLM baselayer prepared for Demeter
                                            in units square-degrees.

    :param clm_spatial_allocation_file:     Full path with file name and extension to
                                            the Demeter spatial allocation file for CLM
                                            landclasses.

    :param out_clm_baselayer_file:          Full path with file name and extension to
                                            to save the reclassified baselayer.

    :method demeter_to_clm_map:             Create a dictionary of Demeter final landclasses
                                            to CLM PFTs. Returns dictionary of
                                            {final_landclass_name: [clm_pft, ...]}}.

    :method get_unique_clm_pfts:            Create a list of unique CLM PFTs represented in the
                                            baselayer.

    :method reclass_clm_baselayer:          Reclassify the existing CLM baselayer for Demeter
                                            to Demeter land cover classes.  Save output to
                                            new file. Returns data frame of reclassified data.

    # accessible upon initialization
    :attribute dem_to_clm_dict:             Dictionary of {final_landclass_name: [clm_pft, ...]}}.

    :attribute df_clm_baselayer:            Data frame of reclassified data.

    """

    def __init__(self, clm_baselayer_file, clm_spatial_allocation_file, out_clm_baselayer_file):

        self.clm_baselayer_file = clm_baselayer_file
        self.clm_spatial_allocation_file = clm_spatial_allocation_file
        self.out_clm_baselayer_file = out_clm_baselayer_file

        # get a dictionary mapping demeter final landcover types to CLM PFTs
        self.dem_to_clm_dict = self.demeter_to_clm_map()

        # reclassify CLM baselayer and save output
        self.df_clm_baselayer = self.reclass_clm_baselayer()

    def demeter_to_clm_map(self, true_value="1"):
        """Create a dictionary of Demeter final landclasses to CLM PFTs.

        :param true_value:                      The value that represents true in the allocation file.

        :return:                                Dictionary of {final_landclass_name: [clm_pft, ...]}}
        """
        # create a dictionary of {clm_pft: [final_landclass_name, ...]}
        d = {}
        with open(self.clm_spatial_allocation_file) as get:
            for ix, line in enumerate(get):
                s = line.strip().split(',')

                # pft name
                pft = s[0]

                # column contents
                col = s[1:]

                if ix == 0:
                    hdr = col
                else:
                    d[pft] = [hdr[idx] for idx, i in enumerate(col) if i == true_value]

        # convert to necessary output format of {final_landclass_name: [clm_pft, ...]}
        do = {}
        for k in d.keys():
            v = d[k][0]

            # append the CLM PFT if not in dict
            if v in do:
                do[v].append(k)
            else:
                do[v] = [k]

        return do

    def get_unique_clm_pfts(self):
        """Create a list of unique CLM PFTs represented in the baselayer.

        :return:                             Unique list of CLM PFTs
        """
        l = []
        for k in self.dem_to_clm_dict.keys():
            l.extend(self.dem_to_clm_dict[k])

        return list(np.unique(np.array(l)))

    def reclass_clm_baselayer(self):
        """Reclassify the existing CLM baselayer for Demeter to Demeter land cover classes.
        Save output to new file.

        :return:                             Data frame of reclassified data
        """
        # read in the CLM baselayer for Demeter
        df = pd.read_csv(self.clm_baselayer_file)

        # for each Demeter land class
        for dem_lc in self.dem_to_clm_dict.keys():
            # get the CLM PFT list for each Demeter land cover type
            clm_cols = self.dem_to_clm_dict[dem_lc]

            # sum CLM PFTs to new Demeter land class column
            df[dem_lc] = df[clm_cols].sum(1)

        # drop CLM PFT columns
        df.drop(self.get_unique_clm_pfts(), axis=1, inplace=True)

        # save reclassified CLM baselayer for Demeter
        df.to_csv(self.out_clm_baselayer_file, index=False)

        return df
