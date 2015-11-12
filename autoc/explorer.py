#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: efourrier

Purpose : This is a framework for Modeling with pandas, numpy and skicit-learn.
The Goal of this module is to rely on a dataframe structure for modelling g

"""


#########################################################
# Import modules and global helpers
#########################################################

import pandas as pd
import numpy as np
from numpy.random import permutation


def cserie(serie):
    return serie[serie].index.tolist()


class DataExploration(object):
    """
    This class is designed to provide infos about the dataset such as
    number of missing values, number of unique values, constant columns,
    long strings ...

    For the most useful methods it will store the result into a attributes

    When you used a method the output will be stored in a instance attribute so you
    don't have to compute the result again.

        """

    def __init__(self, data):
        """
        Parameters
        ----------
        data : DataFrame
                the data you want explore

        Examples
        --------
        explorer = DataExploration(data = your_DataFrame)
        explorer.structure() : global structure of your DataFrame
        explorer.psummary() to get the a global snapchat of the different stuff detected
        data_cleaned = explorer.basic_cleaning() to clean your data.
        """
        assert isinstance(data, pd.DataFrame)
        self.data = data
        # if not self.label:
        # 	print("""the label column is empty the data will be considered
        # 		as a dataset of predictors""")
        self._nrow = len(self.data.index)
        self._ncol = len(self.data.columns)
        self._dfnumi = (self.data.dtypes == float) | (
            self.data.dtypes == int)
        self._dfnum = cserie(self._dfnumi)
        self._dfchari = (self.data.dtypes == object)
        self._dfchar = cserie(self._dfchari)
        self._nacolcount = pd.DataFrame()
        self._narowcount = pd.DataFrame()
        self._count_unique = pd.DataFrame()
        self._constantcol = []
        self._manymissingcol = []
        self._manymissingrow = []
        self._dupcol = []
        self._nearzerovar = pd.DataFrame()
        self._corrcolumns = []
        self._dict_info = {}
        self._structure = pd.DataFrame()
        self._string_info = ""

    # def get_label(self):
    # 	""" return the Serie of label you want predict """
    # 	if not self.label:
    # 		print("""the label column is empty the data will be considered
    # 			as a dataset of predictors""")
    # 	return self.data[self.label]

    def count_unique(self):
        """ Return a serie with the number of unique value per columns """
        if len(self._count_unique):
            return self._count_unique
        self._count_unique = self.data.apply(lambda x: x.nunique(), axis=0)
        return self._count_unique

    def sample_df(self, pct=0.05, nr=10, threshold=None):
        """ sample a number of rows of a dataframe = min(max(0.05*nrow(self,nr),threshold)"""
        a = max(int(pct * float(len(self.data.index))), nr)
        if threshold:
            a = min(a, threshold)
        return self.data.loc[permutation(self.data.index)[:a]]

    def nacolcount(self):
        """ count the number of missing values per columns """
        if len(self._nacolcount):
            return self._nacolcount
        self._nacolcount = self.data.isnull().sum(axis=0)
        self._nacolcount = pd.DataFrame(self._nacolcount, columns=['Nanumber'])
        self._nacolcount['Napercentage'] = self._nacolcount[
            'Nanumber'] / (self._nrow)
        return self._nacolcount

    def narowcount(self):
        """ count the number of missing values per columns """
        if len(self._narowcount):
            return self._narowcount
        self._narowcount = self.data.isnull().sum(axis=1)
        self._narowcount = pd.DataFrame(
            self._narowcount, columns=['Nanumber'])
        self._narowcount['Napercentage'] = self._narowcount[
            'Nanumber'] / (self._nrow)
        return self._narowcount

    def manymissing(self, a=0.9, row=False):
        """ identify columns of a dataframe with many missing values ( >= a), if
        row = True row either.
        - the output is a list """
        if row:
            self._manymissingrow = self.narowcount()
            self._manymissingrow = cserie(
                self._manymissingrow['Napercentage'] >= a)
            return self._manymissingrow
        else:
            self._manymissingcol = self.nacolcount()
            self._manymissingcol = cserie(
                self._manymissingcol['Napercentage'] >= a)
            return self._manymissingcol

    def df_len_string(self):
        """ Return a Series with the max of the length of the string of string-type columns """
        return self.data.drop(self._dfnum, axis=1).apply(lambda x: np.max(x.str.len()), axis=0)

    def detectkey(self, index_format=False, pct=0.15, dropna=False, **kwargs):
        """ identify id or key columns as an index if index_format = True or
        as a list if index_format = False """
        if not dropna:
            col_to_keep = self.sample_df(
                pct=pct, **kwargs).apply(lambda x: len(x.unique()) == len(x), axis=0)
            if len(col_to_keep) == 0:
                return []
            is_key_index = col_to_keep
            is_key_index[is_key_index] == self.data.loc[:, is_key_index].apply(
                lambda x: len(x.unique()) == len(x), axis=0)
            if index_format:
                return is_key_index
            else:
                return cserie(is_key_index)
        else:
            col_to_keep = self.sample_df(
                pct=pct, **kwargs).apply(lambda x: x.nunique() == len(x.dropna()), axis=0)
            if len(col_to_keep) == 0:
                return []
            is_key_index = col_to_keep
            is_key_index[is_key_index] == self.data.loc[:, is_key_index].apply(
                lambda x: x.nunique() == len(x.dropna()), axis=0)
            if index_format:
                return is_key_index
            else:
                return cserie(is_key_index)

    def constantcol(self, **kwargs):
        """ identify constant columns """
        # sample to reduce computation time
        if len(self._constantcol):
            return self._constantcol
        col_to_keep = self.sample_df(
            **kwargs).apply(lambda x: len(x.unique()) == 1, axis=0)
        if len(cserie(col_to_keep)) == 0:
            return []
        self._constantcol = cserie(self.data.loc[:, col_to_keep].apply(
            lambda x: len(x.unique()) == 1, axis=0))
        return self._constantcol

    def constantcol2(self, **kwargs):
        """ identify constant columns """
        return cserie((self.data == self.data.ix[0]).all())

    def factors(self, nb_max_levels=10, threshold_value=None, index=False):
        """ return a list of the detected factor variable, detection is based on
        ther percentage of unicity perc_unique = 0.05 by default.
        We follow here the definition of R factors variable considering that a
        factor variable is a character variable that take value in a list a levels

        this is a bad implementation


        Arguments
        ----------
        nb_max_levels: the mac nb of levels you fix for a categorical variable
        threshold_value : the nb of of unique value in percentage of the dataframe length
        index : if you want the result as an index or a list

         """
        if threshold_value:
            max_levels = max(nb_max_levels, threshold_value * self._nrow)
        else:
            max_levels = nb_max_levels

        def helper_factor(x, num_var=self._dfnum):
            unique_value = set()
            if x.name in num_var:
                return False
            else:
                for e in x.values:
                    if len(unique_value) >= max_levels:
                        return False
                    else:
                        unique_value.add(e)
                return True

        if index:
            return self.data.apply(lambda x: helper_factor(x))
        else:
            return cserie(self.data.apply(lambda x: helper_factor(x)))

    def structure(self, threshold_factor=10):
        """ this function return a summary of the structure of the pandas DataFrame
        data looking at the type of variables, the number of missing values, the
        number of unique values """

        if len(self._structure):
            return self._structure
        dtypes = self.data.dtypes
        nacolcount = self.nacolcount()
        nb_missing = nacolcount.Nanumber
        perc_missing = nacolcount.Napercentage
        nb_unique_values = self.count_unique()
        dtypes_r = self.data.apply(lambda x: "character")
        dtypes_r[self._dfnumi] = "numeric"
        dtypes_r[(dtypes_r == 'character') & (
            nb_unique_values <= threshold_factor)] = 'factor'
        constant_columns = (nb_unique_values == 1)
        na_columns = (perc_missing == 1)
        is_key = nb_unique_values == self._nrow
        # is_key_na = ((nb_unique_values + nb_missing) == self.nrow()) & (~na_columns)
        dict_str = {'dtypes_r': dtypes_r, 'perc_missing': perc_missing,
                    'nb_missing': nb_missing, 'is_key': is_key,
                    'nb_unique_values': nb_unique_values, 'dtypes_p': dtypes,
                    'constant_columns': constant_columns, 'na_columns': na_columns}
        self._structure = pd.concat(dict_str, axis=1)
        self._structure = self._structure.loc[:, ['dtypes_p', 'dtypes_r', 'nb_missing', 'perc_missing',
                                                  'nb_unique_values', 'constant_columns', 'na_columns', 'is_key']]
        return self._structure

    def findupcol(self, threshold=100, **kwargs):
        """ find duplicated columns and return the result as a list of list """
        df_s = self.sample_df(threshold=100, **kwargs).T
        dup_index_s = (df_s.duplicated()) | (
            df_s.duplicated(take_last=True))

        if len(cserie(dup_index_s)) == 0:
            return []

        df_t = (self.data.loc[:, dup_index_s]).T
        dup_index = df_t.duplicated()
        dup_index_complet = cserie(
            (dup_index) | (df_t.duplicated(take_last=True)))

        l = []
        for col in cserie(dup_index):
            index_temp = self.data[dup_index_complet].apply(
                lambda x: (x == self.data[col])).sum() == self._nrow
            temp = list(self.data[dup_index_complet].columns[index_temp])
            l.append(temp)
        self._dupcol = l
        return self._dupcol

    def finduprow(self, subset=[]):
        """ find duplicated rows and return the result a sorted dataframe of all the
        duplicates
        subset is a list of columns to look for duplicates from this specific subset .
        """
        if sum(self.data.duplicated()) == 0:
            print("there is no duplicated rows")
        else:
            if subset:
                dup_index = (self.data.duplicated(subset=subset)) | (
                    self.data.duplicated(subset=subset, take_last=True))
            else:
                dup_index = (self.data.duplicated()) | (
                    self.data.duplicated(take_last=True))

            if subset:
                return self.data[dup_index].sort(subset)
            else:
                return self.data[dup_index].sort(self.data.columns[0])

    def nearzerovar(self, freq_cut=95 / 5, unique_cut=10, save_metrics=False):
        """ identify predictors with near-zero variance.
                freq_cut: cutoff ratio of frequency of most common value to second
                most common value.
                unique_cut: cutoff percentage of unique value over total number of
                samples.
                save_metrics: if False, print dataframe and return NON near-zero var
                col indexes, if True, returns the whole dataframe.
        """
        nb_unique_values = self.count_unique()
        percent_unique = 100 * nb_unique_values / self._nrow

        def helper_freq(x):
            if nb_unique_values[x.name] == 0:
                return 0.0
            elif nb_unique_values[x.name] == 1:
                return 1.0
            else:
                return float(x.value_counts().iloc[0]) / x.value_counts().iloc[1]

        freq_ratio = self.data.apply(helper_freq)

        zerovar = (nb_unique_values == 0) | (nb_unique_values == 1)
        nzv = ((freq_ratio >= freq_cut) & (
            percent_unique <= unique_cut)) | (zerovar)

        if save_metrics:
            return pd.DataFrame({'percent_unique': percent_unique, 'freq_ratio': freq_ratio, 'zero_var': zerovar, 'nzv': nzv}, index=self.data.columns)
        else:
            print(pd.DataFrame({'percent_unique': percent_unique, 'freq_ratio': freq_ratio,
                                'zero_var': zerovar, 'nzv': nzv}, index=self.data.columns))
            return nzv[nzv == True].index

    def findcorr(self, cutoff=.90, method='pearson', data_frame=False, print_mode=False):
        """
        implementation of the Recursive Pairwise Elimination.
        The function finds the highest correlated pair and removes the most
        highly correlated feature of the pair, then repeats the process
        until the threshold 'cutoff' is reached.

        will return a dataframe is 'data_frame' is set to True, and the list
        of predictors to remove oth
        Adaptation of 'findCorrelation' function in the caret package in R.
        """
        res = []
        df = self.data.copy(0)
        cor = df.corr(method=method)
        for col in cor.columns:
            cor[col][col] = 0

        max_cor = cor.max()
        if print_mode:
            print(max_cor.max())
        while max_cor.max() > cutoff:
            A = max_cor.idxmax()
            B = cor[A].idxmax()

            if cor[A].mean() > cor[B].mean():
                cor.drop(A, 1, inplace=True)
                cor.drop(A, 0, inplace=True)
                res += [A]
            else:
                cor.drop(B, 1, inplace=True)
                cor.drop(B, 0, inplace=True)
                res += [B]

            max_cor = cor.max()
            if print_mode:
                print(max_cor.max())

        if data_frame:
            return df.drop(res, 1)
        else:
            return res
            self._corrcolumns = res

    def psummary(self, manymissing_ph=0.70, manymissing_pl=0.05, nzv_freq_cut=95 / 5, nzv_unique_cut=10,
                 threshold=100, string_threshold=40, dynamic=False):
        """
        This function will print you a summary of the dataset, based on function
        designed is this package
        - Output : python print
        It will store the string output and the dictionnary of results in private variables

        """
        nacolcount_p = self.nacolcount().Napercentage
        if dynamic:
            print('there are {0} duplicated rows\n'.format(
                self.data.duplicated().sum()))
            print('the columns with more than {0:.2%} manymissing values:\n{1} \n'.format(manymissing_ph,
                                                                                          cserie((nacolcount_p > manymissing_ph))))

            print('the columns with less than {0:.2%} manymissing values are :\n{1} \n you should fill them with median or most common value \n'.format(
                manymissing_pl, cserie((nacolcount_p > 0) & (nacolcount_p <= manymissing_pl))))

            print('the detected keys of the dataset are:\n{0} \n'.format(
                self.detectkey()))
            print('the duplicated columns of the dataset are:\n{0}\n'.format(
                self.findupcol(threshold=100)))
            print('the constant columns of the dataset are:\n{0}\n'.format(
                self.constantcol()))

            print('the columns with nearzerovariance are:\n{0}\n'.format(
                list(cserie(self.nearzerovar(nzv_freq_cut, nzv_unique_cut, save_metrics=True).nzv))))
            print('the columns highly correlated to others to remove are:\n{0}\n'.format(
                self.findcorr(data_frame=False)))
            print('these columns contains big strings :\n{0}\n'.format(
                cserie(self.df_len_string() > string_threshold)))
        else:
            self._dict_info = {'nb_duplicated_rows': sum(self.data.duplicated()),
                               'many_missing_percentage': manymissing_ph,
                               'manymissing_columns': cserie((nacolcount_p > manymissing_ph)),
                               'low_missing_percentage': manymissing_pl,
                               'lowmissing_columns': cserie((nacolcount_p > 0) & (nacolcount_p <= manymissing_pl)),
                               'keys_detected': self.detectkey(),
                               'dup_columns': self.findupcol(threshold=100),
                               'constant_columns': self.constantcol(),
                               'nearzerovar_columns': cserie(self.nearzerovar(nzv_freq_cut, nzv_unique_cut, save_metrics=True).nzv),
                               'high_correlated_col': self.findcorr(data_frame=False),
                               'big_strings_col': cserie(self.df_len_string() > string_threshold)
                               }

            self._string_info = u"""
		there are {nb_duplicated_rows} duplicated rows\n
		the columns with more than {many_missing_percentage:.2%} manymissing values:\n{manymissing_columns} \n
		the columns with less than {low_missing_percentage:.2%}% manymissing values are :\n{lowmissing_columns} \n
		you should fill them with median or most common value\n
		the detected keys of the dataset are:\n{keys_detected} \n
		the duplicated columns of the dataset are:\n{dup_columns}\n
		the constant columns of the dataset are:\n{constant_columns}\n
		the columns with nearzerovariance are:\n{nearzerovar_columns}\n
		the columns highly correlated to others to remove are:\n{high_correlated_col}\n
		these columns contains big strings :\n{big_strings_col}\n
				""".format(**self._dict_info)
            print(self._string_info)
