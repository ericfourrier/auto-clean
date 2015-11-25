from autoc.explorer import DataExploration
from autoc.utils.helpers import cserie

class NaImputer(DataExploration):

    def __init__(self, *args, **kwargs):
        super(NaImputer, self).__init__(*args, **kwargs)

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
