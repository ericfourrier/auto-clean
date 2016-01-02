from autoc.explorer import DataExploration, pd
from autoc.utils.helpers import cserie
import seaborn as sns
import matplotlib.pyplot as plt
from autoc.utils.helpers import cached_property
from autoc.utils.coorplot import plot_corrmatrix

def missing_map(df, nmax=100, verbose=True, yticklabels=False, figsize=(15, 11), *args, **kwargs):
    """ Returns missing map plot like in amelia 2 package in R """
    f, ax = plt.subplots(figsize=figsize)
    if nmax < df.shape[0]:
        df_s = df.sample(n=nmax)  # sample rows if dataframe too big
    return sns.heatmap(df_s.isnull(), yticklabels=yticklabels, vmax=1, *args, **kwargs)

class NaImputer(DataExploration):

    def __init__(self, *args, **kwargs):
        super(NaImputer, self).__init__(*args, **kwargs)

    def infos_na(self, na_low=0.05, na_high=0.90):
        """ Returns a dict with various infos about missing values """
        infos = {}
        infos['nacolcount'] = self.nacolcount()
        infos['narowcount'] = self.narowcount()
        infos['nb_total_na'] = self.total_missing
        infos['many_na_col'] = self.manymissing(pct=na_high)
        infos['low_na_col'] = self.nacolcount().Napercentage < na_low
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

    @cached_property
    def data_isna(self, prefix="is_na_", subset=None):
        """ Returns dataset with is_na columns from the a dataframe with missing values """
        if subset is not None:
            data_isna = self.data[subset].apply(lambda x: self.get_isna(x.name))
        else:
            data_isna = self.data.apply(lambda x: self.get_isna(x.name))
        data_isna.columns = ["{}{}".format(prefix, c) for c in data_isna.columns]
        return data_isna

    def corrplot_na(self, remove_cst_col=True, *args, **kwargs):
        """ Returns a corrplot of data_isna """
        if remove_cst_col:
            plot_corrmatrix(self.data_isna.loc[:, self.data_isna.var() > 0], *args, **kwargs)
        else:
            plot_corrmatrix(self.data_isna, *args, **kwargs)

    def get_isna_mean(self, colname, prefix="is_na_"):
        """ Returns empirical conditional expectatation, std, and sem of other numerical variable
        for a certain colname with 0:not_a_na 1:na """
        na_colname = "{}{}".format(prefix, colname)
        cols_to_keep = list(self.data.columns) + [na_colname]
        functions = ['mean', 'std', 'sem']
        return self.data_isna_m.loc[:, cols_to_keep].groupby(na_colname).agg(functions).transpose()

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
            print("There is {0:.2%} rows matching conditions".format(pct_missing))
        if not index:
            return self.data.loc[~index_missing, :]
        else:
            return index_missing


    @staticmethod
    def fillna_serie(serie, special_value=None):
        """ fill values in a serie default with the mean for numeric or the most common
        factor for categorical variable"""
        if special_value is not None:
            return serie.fillna(special_value)  # "Missing for example"
        elif (serie.dtype == float) | (serie.dtype == int):
            return serie.fillna(serie.median())  # fill with median
        else:
            # Fill with most common value
            return serie.fillna(serie.value_counts().index[0])

    def basic_naimputation(self, columns_to_process=[], threshold=None):
        """ this function will return a dataframe with na value replaced int
        the columns selected by the mean or the most common value

        Arguments
        ---------
        - columns_to_process : list of columns name with na values you wish to fill
        with the fillna_serie function

        Returns
        --------
        - a pandas Dataframe with the columns_to_process filled with the filledna_serie

         """
        df = self.data
        if threshold:
            columns_to_process=columns_to_process + cserie(self.nacolcount().Napercentage < threshold)
        df.loc[:, columns_to_process]=df.loc[:, columns_to_process].apply(lambda x: self.fillna_serie(x))
        return df
