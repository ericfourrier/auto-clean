from autoc.explorer import DataExploration, pd
from autoc.utils.helpers import cserie
import seaborn as sns
import matplotlib.pyplot as plt
#from autoc.utils.helpers import cached_property
from autoc.utils.corrplot import plot_corrmatrix
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats.mstats import ks_2samp

def missing_map(df, nmax=100, verbose=True, yticklabels=False, figsize=(15, 11), *args, **kwargs):
    """ Returns missing map plot like in amelia 2 package in R """
    f, ax = plt.subplots(figsize=figsize)
    if nmax < df.shape[0]:
        df_s = df.sample(n=nmax)  # sample rows if dataframe too big
    return sns.heatmap(df_s.isnull(), yticklabels=yticklabels, vmax=1, *args, **kwargs)

# class ColumnNaInfo


class NaImputer(DataExploration):

    def __init__(self, *args, **kwargs):
        super(NaImputer, self).__init__(*args, **kwargs)
        self.get_data_isna()

    @property
    def nacols(self):
        """ Returns a list of column with at least one missing values """
        return cserie(self.nacolcount().Nanumber > 0)

    @property
    def nacols_i(self):
        """ Returns the index of column with at least one missing values """
        return cserie(self.nacolcount().Nanumber > 0)

    def get_overlapping_matrix(self, normalize=True):
        """ Look at missing values overlapping """
        arr = self.data_isna.astype('float').values
        arr = np.dot(arr.T, arr)
        if normalize:
            arr = arr / (arr.max(axis=1)[:, None])
        index = self.nacols
        res = pd.DataFrame(index=index, data=arr, columns=index)
        return res

    def infos_na(self, na_low=0.05, na_high=0.90):
        """ Returns a dict with various infos about missing values """
        infos = {}
        infos['nacolcount'] = self.nacolcount()
        infos['narowcount'] = self.narowcount()
        infos['nb_total_na'] = self.total_missing
        infos['many_na_col'] = self.manymissing(pct=na_high)
        infos['low_na_col'] = cserie(self.nacolcount().Napercentage < na_low)
        infos['total_pct_na'] = self.nacolcount().Napercentage.mean()
        return infos

    def get_isna(self, col):
        """ Returns a dummy variable indicating in a observation of a specific col
            is na or not 0 -> not na , 1 -> na """
        return self.data.loc[:, col].isnull().astype(int)

    @property
    def data_isna_m(self):
        """ Returns merged dataframe (data, data_is_na)"""
        return pd.concat((self.data, self.data_isna), axis=1)

    def get_data_isna(self, prefix="is_na_", filter_nna=True):
        """ Returns dataset with is_na columns from the a dataframe with missing values

        Parameters
        ----------
        prefix : str
            the name of the prefix that will be append to the column name.
        filter_nna: bool
            True if you want remove column without missing values.
        """
        if not filter_nna:
            cols_to_keep = self.data.columns
        else:
            cols_to_keep = self.nacols
        data_isna = self.data.loc[:, cols_to_keep].isnull().astype(int)
        data_isna.columns = ["{}{}".format(prefix, c) for c in cols_to_keep]
        self.data_isna = data_isna
        return self.data_isna

    def get_corrna(self, *args, **kwargs):
        """ Get matrix of correlation of na """
        return self.data_isna.corr(*args, **kwargs)

    def corrplot_na(self, *args, **kwargs):
        """ Returns a corrplot of data_isna """
        print("This function is deprecated")
        plot_corrmatrix(self.data_isna, *args, **kwargs)

    def plot_corrplot_na(self, *args, **kwargs):
        """ Returns a corrplot of data_isna """
        plot_corrmatrix(self.data_isna, *args, **kwargs)

    def plot_density_m(self, colname, subset=None, prefix="is_na_", size=6, *args, **kwargs):
        """ Plot conditionnal density plot from all columns or subset based on
        is_na_colname 0 or 1"""
        colname_na = prefix + colname
        density_columns = self.data.columns if subset is None else subset
        # filter only numeric values and different values from is_na_col
        density_columns = [c for c in density_columns if (
            c in self._dfnum and c != colname)]
        print(density_columns)
        for col in density_columns:
            g = sns.FacetGrid(data=self.data_isna_m, col=colname_na, hue=colname_na,
                              size=size, *args, **kwargs)
            g.map(sns.distplot, col)

    def get_isna_mean(self, colname, prefix="is_na_"):
        """ Returns empirical conditional expectatation, std, and sem of other numerical variable
        for a certain colname with 0:not_a_na 1:na """
        na_colname = "{}{}".format(prefix, colname)
        cols_to_keep = list(self.data.columns) + [na_colname]
        measure_var = self.data.columns.tolist()
        measure_var = [c for c in measure_var if c != colname]
        functions = ['mean', 'std', 'sem']
        return self.data_isna_m.loc[:, cols_to_keep].groupby(na_colname)[measure_var].agg(functions).transpose()

    def get_isna_ttest_s(self, colname_na, colname, type_test="ks"):
        """ Returns tt test for colanme-na and a colname  """
        index_na = self.data.loc[:, colname_na].isnull()
        measure_var = self.data.loc[:, colname].dropna()  # drop na vars
        if type_test == "ttest":
            return ttest_ind(measure_var[index_na], measure_var[~index_na])
        elif type_test == "ks":
            return ks_2samp(measure_var[index_na], measure_var[~index_na])

    def get_isna_ttest(self, colname_na, type_test="ks"):
        res = pd.DataFrame()
        col_to_compare = [c for c in self._dfnum if c !=
                          colname_na]  # remove colname_na
        for col in col_to_compare:
            ttest = self.get_isna_ttest_s(colname_na, col, type_test=type_test)
            res.loc[col, 'pvalue'] = ttest[1]
            res.loc[col, 'statistic'] = ttest[0]
            res.loc[col, 'type_test'] = type_test
        return res

    def isna_summary(self, colname, prefix="is_na_"):
        """ Returns summary from one col with describe """
        na_colname = "{}{}".format(prefix, colname)
        cols_to_keep = list(self.data.columns) + [na_colname]
        return self.data_isna_m.loc[:, cols_to_keep].groupby(na_colname).describe().transpose()

    def delete_narows(self, pct, index=False):
        """ Delete rows with more na percentage than > perc in data
        Return the index

        Arguments
        ---------
        pct : float
            percentage of missing values, rows with more na percentage
            than > perc are deleted
        index : bool, default False
            True if you want an index and not a Dataframe
        verbose : bool, default False
            True if you want to see percentage of data discarded

        Returns
        --------
        - a pandas Dataframe with rows deleted if index=False, index of
        columns to delete either
        """
        index_missing = self.manymissing(pct=pct, axis=0, index=False)
        pct_missing = len(index_missing) / len(self.data.index)
        if verbose:
            print("There is {0:.2%} rows matching conditions".format(
                pct_missing))
        if not index:
            return self.data.loc[~index_missing, :]
        else:
            return index_missing

    def fillna_serie(self, colname, threshold_factor=0.1, special_value=None, date_method='ffill'):
        """ fill values in a serie default with the mean for numeric or the most common
        factor for categorical variable"""
        if special_value is not None:
            # "Missing for example"
            return self.data.loc[:, colname].fillna(special_value)
        elif self.data.loc[:, colname].dtype == float:
            # fill with median
            return self.data.loc[:, colname].fillna(self.data.loc[:, colname].median())
        elif self.is_int_factor(colname, threshold_factor):
            return self.data.loc[:, colname].fillna(self.data.loc[:, colname].mode()[0])
        # fillna for datetime with the method provided by pandas
        elif self.data.loc[:, colname].dtype == '<M8[ns]':
            return self.data.loc[:, colname].fillna(method=date_method)
        else:
            # Fill with most common value
            return self.data.loc[:, colname].fillna(self.data.loc[:, colname].value_counts().index[0])

    def basic_naimputation(self, columns_to_process=[], threshold=None):
        """ this function will return a dataframe with na value replaced int
        the columns selected by the mean or the most common value

        Arguments
        ---------
        - columns_to_process : list of columns name with na values you wish to fill
        with the fillna_serie function

        Returns
        --------
        - a pandas DataFrame with the columns_to_process filled with the fillena_serie

        """

        # self.data = self.df.copy()
        if threshold:
            columns_to_process = columns_to_process + cserie(self.nacolcount().Napercentage < threshold)
        self.data.loc[:, columns_to_process] = self.data.loc[
            :, columns_to_process].apply(lambda x: self.fillna_serie(colname=x.name))
        return self.data

    def split_tt_na(self, colname, index=False):
        """ Split the dataset returning the index of test , train """
        index_na = self.data.loc[:, colname].isnull()
        index_test = (index_na == True)
        index_train = (index_na == False)
        if index:
            return index_test, index_train
        else:
            return self.data.loc[index_test, :], self.data.loc[index_train, :]
