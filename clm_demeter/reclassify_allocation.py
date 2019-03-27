import pandas as pd


__author__ = 'Chris R. Vernon'
__email__ = 'chris.vernon@pnnl.gov'
__copyright__ = 'Copyright (c) 2017, Battelle Memorial Institute'
__license__ = 'BSD 2-Clause'


def reclass_spatial_allocation(spatial_allocation_file, out_spatial_allocation_file):
    """Reclassify the spatial allocation file for Demeter to 1:1 land class relationships
    for Demeter. Using the output of this functions assumes that the user has already
    created a reclassified base layer.

    :param spatial_allocation_file:     Full path with file name and extension to
                                        the Demeter spatial allocation file for CLM
                                        landclasses.

    :param out_spat_allocation_file:    Full path with file name and extension for the reclassified spatial allocation
                                        file.

    """
    base_alloc = pd.read_csv(spatial_allocation_file)

    # get a list of Demeter final land classes
    dem_lcs = base_alloc.columns[1:].tolist()

    # create list of zeros to populate allocation
    alloc_list = ['0'] * len(dem_lcs)

    with open(out_spatial_allocation_file, 'w') as out:
        # write header
        out.write('{}\n'.format(','.join(base_alloc.columns)))

        for index, lc in enumerate(dem_lcs):
            # copy alloc list and target lcs value to 1
            lx = alloc_list.copy()
            lx[index] = '1'

            out.write('{0},{1}\n'.format(dem_lcs[index], ','.join(lx)))

    return


def reclass_projected_allocation(projected_data_file, projected_allocation_file, out_proj_allocation_file):
    """Reclassify the projected allocation file for Demeter to account for classes that have
    been disaggregated to Demeter's final land classes.

    :param projected_file:              Full path with file name and extension of the projected data file that has been
                                        extracted from a GCAM output database for use with Demeter.

    :param projected_allocation_file:   Full path with file name and extension of the projected allocation file from
                                        Demeter that maps GCAM land classes to the final land cover classes in
                                        Demeter.

    :param out_proj_allocation_file:    Full path with file name and extension for the reclassified projected allocation
                                        file.

    """
    df = pd.read_csv(projected_data_file)

    proj_lcs = df['landclass'].unique()

    alloc = pd.read_csv(projected_allocation_file)

    # get a list of Demeter final land classes
    dem_lcs = alloc.columns[1:].tolist()

    # create list of zeros to populate allocation
    alloc_list = ['0'] * len(dem_lcs)

    for i in proj_lcs:
        if i not in dem_lcs:
            print("Projected land class '{}' not in Demeter land classes.".format(i))

    with open(out_proj_allocation_file, 'w') as out:

        # write header
        out.write('{}\n'.format(','.join(alloc.columns)))

        for index, lc in enumerate(dem_lcs):

            if lc not in proj_lcs:
                print("Demeter land class '{}' not in projected data file.".format(lc))

            # copy alloc list and target lcs value to 1
            lx = alloc_list.copy()
            lx[index] = '1'

            out.write('{0},{1}\n'.format(dem_lcs[index], ','.join(lx)))

    return
