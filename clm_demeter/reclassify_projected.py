import numpy as np
import pandas as pd


__author__ = 'Chris R. Vernon'
__email__ = 'chris.vernon@pnnl.gov'
__copyright__ = 'Copyright (c) 2017, Battelle Memorial Institute'
__license__ = 'BSD 2-Clause'


class GcamLandclassSplit:
    """Split a GCAM landclass into multiple classes based on the fractional amount present in the observed data per
    subregion.  This method is more desirable than the default "even percentage" split that Demeter conducts.  The
    output file replaces the GCAM target landclass (e.g. RockIceDesert) with the user-selected classes (e.g. snow and
    sparse) per subregion.  The new file becomes what is referenced as the projected file in Demeter.

    :param observed_file:               Full path with file name and extension of the observed data file to be used in the
                                        Demeter run. Optionally pass a Pandas Data Frame.

    :param projected_file:              Full path with file name and extension of the projected data file that has been
                                        extracted from a GCAM output database for use with Demeter.
                                        Optionally pass a Pandas Data Frame.

    :param target_landclass:            Name of the landclass from the projected file to split (e.g. RockIceDesert).

    :param observed_landclasses:        List of landclass names from the observed data to substitute (e.g. ['snow', 'sparse'].

    :param metric:                      Name of the subregion used. Either 'basin_id' or 'aez_id'.

    :param gcam_year_list:              List of GCAM years to process.

    :param out_file:                    Full path with file name and extension for the altered projected data file.

    :return:                            Data frame; save as file

    """
    # region id field name used by Demeter
    REGION_NAME_FIELD = 'region'
    REGION_ID_FIELD = 'region_id'
    PRJ_METRIC_ID_FIELD = 'metric_id'
    PRJ_LANDCLASS_FIELD = 'landclass'
    UNIT_FIELD = 'Units'

    def __init__(self, observed_file, projected_file, target_landclass, observed_landclasses, metric,
                 gcam_year_list, out_file=None):

        self.observed_file = observed_file
        self.projected_file = projected_file
        self.target_landclass = target_landclass
        self.observed_landclasses = observed_landclasses
        self.metric = metric
        self.gcam_year_list = [str(i) for i in gcam_year_list]
        self.out_file = out_file

        # name of the combined region and subregion field
        self.subregion_field = 'reg_{}'.format(self.metric.split('_')[0])

        # disaggregate landclass
        self.df = self.disaggregate_landclass()

    def calc_observed_fraction(self):
        """Calculate the fraction of land area for each target observed landclass per subregion.

        :return:                    Dictionary; {region_metric_id: (obs_lc, ...), ...}

        """
        cols = [GcamLandclassSplit.REGION_ID_FIELD, self.metric]
        cols.extend(self.observed_landclasses)

        err_msg = "WARNING: One of the target projected landclasses '{}' in projected data is not in the observed data."

        if type(self.observed_file) == pd.core.frame.DataFrame:
            # get a list of expected land classes not in the observed file
            ck = [i for i in self.observed_landclasses if i not in self.observed_file.columns]

            if len(ck) > 0:
                print(err_msg.format(ck))
                return None, None

            else:
                df = self.observed_file[cols].copy()

        else:
            try:
                df = pd.read_csv(self.observed_file, usecols=cols)
            except ValueError:
                print(err_msg.format(cols))
                return None, None

        # get total amount of observed landclasses in each subregion
        gdf = df.groupby([GcamLandclassSplit.REGION_ID_FIELD, self.metric]).sum(axis=1)
        gdf.reset_index(inplace=True)

        # create key
        gdf[self.subregion_field] = gdf[GcamLandclassSplit.REGION_ID_FIELD].astype(str) + '_' + gdf[self.metric].astype(
            str)

        # sum observed landclasses (e.g., snow + sparse) per subregion
        gdf['total'] = gdf[self.observed_landclasses].sum(axis=1)

        # create fractional value of each observed landclass per subregion
        frac_lcs = []
        for lc in self.observed_landclasses:
            frac_lc = 'frac_{}'.format(lc)
            frac_lcs.append(frac_lc)

            gdf[frac_lc] = np.where(gdf[lc] / gdf['total'] > 1, 1, gdf[lc] / gdf['total'])

        drop_cols = self.observed_landclasses + ['total', GcamLandclassSplit.REGION_ID_FIELD, self.metric]
        gdf.drop(drop_cols, axis=1, inplace=True)

        # fill subregions that have none of the target values with 0
        gdf.fillna(0, inplace=True)

        return gdf, frac_lcs

    def disaggregate_landclass(self):
        """Disaggregate GCAM target landclass using observed data fraction.  Save output to file.

        :return:                    Data frame

        """

        # get the fractional area from each landclass from the observed data
        obs_df, frac_lcs = self.calc_observed_fraction()

        # if a data frame is being passed instead of a file path
        if type(self.projected_file) == pd.core.frame.DataFrame:
            prj_df = self.projected_file.copy()
        else:
            prj_df = pd.read_csv(self.projected_file)

        # if the target field from the projected data is not in the observed data, return as-is
        if obs_df is None:

            prj_df.to_csv(self.out_file, index=False)

            # if there is more than 1 land class, return as-is; else, rename land class and return
            if len(self.observed_landclasses) > 1:
                return prj_df

            else:
                prj_df[GcamLandclassSplit.PRJ_LANDCLASS_FIELD] = prj_df[
                    GcamLandclassSplit.PRJ_LANDCLASS_FIELD].str.replace(
                    self.target_landclass, self.observed_landclasses[0])

            if self.out_file is not None:
                prj_df.to_csv(self.out_file, index=False)

            return prj_df

        # add region_metric field
        prj_df[self.subregion_field] = prj_df[GcamLandclassSplit.REGION_ID_FIELD].astype(str) + '_' + prj_df[
            GcamLandclassSplit.PRJ_METRIC_ID_FIELD].astype(str)

        # join fractional fields from observed data
        prj_df = pd.merge(prj_df, obs_df, on=self.subregion_field, how='left')

        # data frame containing only the target landclass records
        lc_df = prj_df.loc[prj_df[GcamLandclassSplit.PRJ_LANDCLASS_FIELD] == self.target_landclass].copy()

        # get a list of all fractional columns
        frac_cols = [i for i in lc_df.columns if 'frac_' in i]

        # sum fractional columns to account for subregions in the base data that have no matching area
        lc_df['frac_check'] = lc_df[frac_cols].sum(axis=1)

        # if there is not corresponding land in the base layer subregion, split projection evenly between classes
        # get fractional amount from even split
        disperse_frac = 1.0 / len(frac_cols)

        for fc in frac_cols:
            lc_df[fc] = np.where(lc_df['frac_check'] == 0, disperse_frac, lc_df[fc])

        # drop check field
        lc_df.drop('frac_check', axis=1, inplace=True)

        # data frame containing ALL but the target landclass records
        out_df = prj_df.loc[prj_df[GcamLandclassSplit.PRJ_LANDCLASS_FIELD] != self.target_landclass].copy()

        for lc in self.observed_landclasses:
            idf = lc_df.copy()

            # set landclass to new field name
            idf[GcamLandclassSplit.PRJ_LANDCLASS_FIELD] = lc

            for yr in self.gcam_year_list:
                idf[yr] *= idf['frac_{}'.format(lc)]

            # add new outputs to data frame
            out_df = pd.concat([idf, out_df])

        # drop processing columns
        frac_lcs.append(self.subregion_field)
        out_df.drop(frac_lcs, axis=1, inplace=True)

        # sum like rows
        out_df = out_df.groupby([GcamLandclassSplit.REGION_NAME_FIELD,
                                 GcamLandclassSplit.REGION_ID_FIELD,
                                 GcamLandclassSplit.PRJ_METRIC_ID_FIELD,
                                 GcamLandclassSplit.PRJ_LANDCLASS_FIELD,
                                 GcamLandclassSplit.UNIT_FIELD]).sum(axis=1)

        out_df.reset_index(inplace=True)

        if self.out_file is not None:
            out_df.to_csv(self.out_file, index=False)

        return out_df


def gcam_to_demeter_lc_dict(projected_allocation_file):
    """Create a dictionary of GCAM land classes to Demeter land classes from the
     Demeter projected allocation file.

    :param projected_allocation_file:   Full path with file name and extension of the projected allocation file from
                                        Demeter that maps GCAM land classes to the final land cover classes in
                                        Demeter.

    :return:            Dictionary of {gcam_lcs: [demeter_lc, ...]}

    """
    d = {}
    with open(projected_allocation_file) as get:
        for ix, line in enumerate(get):
            s = line.strip().split(',')
            pft = s[0]
            col = s[1:]

            if ix == 0:
                hdr = col
            else:
                d[pft] = [hdr[idx] for idx, i in enumerate(col) if i == "1"]
    return d


def batch_process_split(projected_allocation_file, observed_baselayer_file, projected_file, out_projected_file,
                        metric, gcam_year_list):
    """Batch process projected land class disaggregation.

    :param projected_allocation_file:   Full path with file name and extension of the projected allocation file from
                                        Demeter that maps GCAM land classes to the final land cover classes in
                                        Demeter.

    :param observed_file:               Full path with file name and extension of the observed data file to be used in the
                                        Demeter run.

    :param projected_file:              Full path with file name and extension of the projected data file that has been
                                        extracted from a GCAM output database for use with Demeter.

    :param out_projected_file:          Full path with file name and extension to save the modified projected file to.

    :param metric:                      Name of the subregion used. Either 'basin_id' or 'aez_id'.

    :param gcam_year_list:              List of GCAM years to process.

    :return:                            Modified projected data frame.

    """
    gcam_dict = gcam_to_demeter_lc_dict(projected_allocation_file)

    last_iteration = len(gcam_dict.keys()) - 1

    for index, gcam_lc in enumerate(gcam_dict.keys()):

        dem_lc_list = gcam_dict[gcam_lc]

        print("Disaggregating projected land class '{}' to '{}'".format(gcam_lc, dem_lc_list))

        # pass file name if first iteration, else pass data frame
        if index == 0:
            g = GcamLandclassSplit(observed_baselayer_file,
                                   projected_file,
                                   gcam_lc,
                                   dem_lc_list,
                                   metric,
                                   gcam_year_list,
                                   out_file=None)

        elif index == last_iteration:
            GcamLandclassSplit(observed_baselayer_file,
                               g.df,
                               gcam_lc,
                               dem_lc_list,
                               metric,
                               gcam_year_list,
                               out_file=out_projected_file)

        else:
            g = GcamLandclassSplit(observed_baselayer_file,
                                   g.df,
                                   gcam_lc,
                                   dem_lc_list,
                                   metric,
                                   gcam_year_list,
                                   out_file=None)

    return g.df
