"""
@author: efourrier

Purpose : This is a simple experimental class to detect outliers. This class
can be used to detect missing values encoded as outlier (-999, -1, ...)


"""

from autoc.explorer import DataExploration, pd
import numpy as np
#from autoc.utils.helpers import cserie
from exceptions import NotNumericColumn


def iqr(ndarray):
    return np.percentile(ndarray, 75) - np.percentile(ndarray, 25)


def z_score(ndarray):
    return (ndarray - np.mean(ndarray)) / (np.std(ndarray))


def iqr_score(ndarray):
    return (ndarray - np.median(ndarray)) / (iqr(ndarray))


def mad_score(ndarray):
    return (ndarray - np.median(ndarray)) / (np.median(np.absolute(ndarray - np.median(ndarray))) / 0.6745)


class OutliersDetection(DataExploration):
    """
    this class focuses on identifying outliers

    Parameters
    ----------
    data : DataFrame

    Examples
    --------
    * od = OutliersDetection(data = your_DataFrame)
    * cleaner.structure() : global structure of your DataFrame
    """

    def __init__(self, *args, **kwargs):
        super(OutliersDetection, self).__init__(*args, **kwargs)
        self.strong_cutoff = {'cutoff_z': 6,
                              'cutoff_iqr': 6, 'cutoff_mad': 6}
        self.basic_cutoff = {'cutoff_z': 3,
                             'cutoff_iqr': 2, 'cutoff_mad': 2}


    def check_negative_value(self, colname):
        """ this function will detect if there is at leat one
         negative value and calculate the ratio negative postive/
        """
        if not self.is_numeric(colname):
            NotNumericColumn("The serie should be numeric values")
        return sum(serie < 0)

    def outlier_detection_serie_1d(self, colname, cutoff_params, scores=[z_score, iqr_score, mad_score]):
        if not self.is_numeric(colname):
            raise("auto-clean doesn't support outliers detection for Non numeric variable")
        keys = [str(func.__name__) for func in scores]
        df = pd.DataFrame(dict((key, func(self.data.loc[:, colname]))
                               for key, func in zip(keys, scores)))
        df['is_outlier'] = 0
        for s in keys:
            cutoff_colname = "cutoff_{}".format(s.split('_')[0])
            index_outliers = np.absolute(df[s]) >= cutoff_params[cutoff_colname]
            df.loc[index_outliers, 'is_outlier'] = 1
        # if 'z_score' in keys:
        #     df.loc[np.absolute(df['z_score']) >=
        #            cutoff_params["cutoff_z"], 'is_outlier'] = 1
        # if 'iqr_score' in keys:
        #     df.loc[np.absolute(df['iqr_score']) >=
        #            cutoff_params["cutoff_iqr"], 'is_outlier'] = 1
        # if 'mad_score' in keys:
        #     df.loc[np.absolute(df['mad_score']) >=
        #            cutoff_params["cutoff_mad"], 'is_outlier'] = 1
        return df

    def check_negative_value(self):
        """ this will return a the ratio negative/positve for each numeric
        variable of the DataFrame
        """
        return self.data[self._dfnum].apply(lambda x: self.check_negative_value_serie(x.name))

    def outlier_detection_1d(self, cutoff_params, subset=None,
                             scores=[z_score, iqr_score, mad_score]):
        """ Return a dictionnary with z_score,iqr_score,mad_score as keys and the
        associate dataframe of distance as value of the dictionnnary"""
        df = self.data.copy()
        numeric_var = self._dfnum
        if subset:
            df = df.drop(subset, axis=1)
        df = df.loc[:, numeric_var]  # take only numeric variable
        # if remove_constant_col:
        # df = df.drop(self.constantcol(), axis = 1) # remove constant variable
        df_outlier = pd.DataFrame()
        for col in df:
            df_temp = self.outlier_detection_serie_1d(col, scores, cutoff_params)
            df_temp.columns = [col + '_' +
                               col_name for col_name in df_temp.columns]
            #df_outlier = pd.concat([df_outlier, df_temp], axis=1)
        return df_temp
