#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: efourrier

Purpose : The purpose of this class is too automaticely transfrom a DataFrame
into a numpy ndarray in order to use an aglorithm

"""


#########################################################
# Import modules and global helpers
#########################################################

from autoc.explorer import DataExploration, pd
import numpy as np
from numpy.random import permutation
from autoc.utils.helpers import cserie
from autoc.exceptions import NumericError




class PreProcessor(DataExploration):
    subtypes = ['text_raw', 'text_categorical', 'ordinal', 'binary', 'other']

    def __init__(self, *args, **kwargs):
        super(PreProcessor, self).__init__(*args, **kwargs)
        self.long_str_cutoff = 80
        self.short_str_cutoff = 30
        self.perc_unique_cutoff = 0.2
        self.nb_max_levels = 20

    def basic_cleaning(self,filter_nacols=True, drop_col=None,
                       filter_constantcol=True, filer_narows=True,
                       verbose=True, filter_rows_duplicates=True, inplace=False):
        """
        Basic cleaning of the data by deleting manymissing columns,
        constantcol, full missing rows,  and drop_col specified by the user.
        """


        col_to_remove = []
        index_to_remove = []
        if filter_nacols:
            col_to_remove += self.nacols_full
        if filter_constantcol:
            col_to_remove += list(self.constantcol())
        if filer_narows:
            index_to_remove += cserie(self.narows_full)
        if filter_rows_duplicates:
            index_to_remove += cserie(self.data.duplicated())
        if isinstance(drop_col, list):
            col_to_remove += drop_col
        elif isinstance(drop_col, str):
            col_to_remove += [drop_col]
        else:
            pass
        col_to_remove = list(set(col_to_remove))
        index_to_remove = list(set(index_to_remove))
        if verbose:
            print("We are removing the folowing columns : {}".format(col_to_remove))
            print("We are removing the folowing rows : {}".format(index_to_remove))
        if inplace:
            return self.data.drop(index_to_remove).drop(col_to_remove, axis=1)
        else:
            return self.data.copy().drop(index_to_remove).drop(col_to_remove, axis=1)

    def _infer_subtype_col(self, colname):
        """ This fonction tries to infer subtypes in order to preprocess them
        better for skicit learn. You can find the different subtypes in the class
        variable subtypes

        To be completed ....
        """
        serie_col = self.data.loc[:, colname]
        if serie_col.nunique() == 2:
            return 'binary'
        elif serie_col.dtype.kind == 'O':
            if serie_col.str.len().mean()  > self.long_str_cutoff and serie_col.nunique()/len(serie_col) > self.perc_unique_cutoff:
                return "text_long"
            elif serie_col.str.len().mean()  <= self.short_str_cutoff and serie_col.nunique() <= self.nb_max_levels:
                return 'text_categorical'
        elif self.is_numeric(colname):
            if serie_col.dtype == int and serie_col.nunique() <= self.nb_max_levels:
                return "ordinal"
        else :
            return "other"

    def infer_subtypes(self):
        """ Apply _infer_subtype_col to the whole DataFrame as a dictionnary  """
        return {col: {'dtype': self.data.loc[:,col].dtype, 'subtype':self._infer_subtype_col(col)} for col in self.data.columns}


    def infer_categorical_str(self, colname,  nb_max_levels=10, threshold_value=0.01):
        """ Returns True if we detect in the serie a  factor variable
        A string factor is based on the following caracteristics :
        ther percentage of unicity perc_unique = 0.05 by default.
        We follow here the definition of R factors variable considering that a
        factor variable is a character variable that take value in a list a levels

        Arguments
        ----------
        nb_max_levels: int
            the max nb of levels you fix for a categorical variable
        threshold_value : float
        the nb of of unique value in percentage of the dataframe length
        """
        # False for numeric columns
        if threshold_value:
            max_levels = max(nb_max_levels, threshold_value * self._nrow)
        else:
            max_levels = nb_max_levels
        if self.is_numeric(colname):
            return False
        # False for categorical columns
        if self.data.loc[:, colname].dtype == "category":
            return False
        unique_value = set()
        for i, v in self.data.loc[:, colname], iteritems():
            if len(unique_value) >= max_levels:
                return False
            else:
                unique_value.add(v)
        return True

    def get_factors(self, nb_max_levels=10, threshold_value=None, index=False):
        """ Return a list of the detected factor variable, detection is based on
        ther percentage of unicity perc_unique = 0.05 by default.
        We follow here the definition of R factors variable considering that a
        factor variable is a character variable that take value in a list a levels

        this is a bad implementation


        Arguments
        ----------
        nb_max_levels: int
            the max nb of levels you fix for a categorical variable.
        threshold_value : float
            the nb of of unique value in percentage of the dataframe length.
        index: bool
            False, returns a list, True if you want an index.


        """
        res = self.data.apply(lambda x: self.infer_categorical_str(x))
        if index:
            return res
        else:
            return cserie(res)

    def factors_to_categorical(self, inplace=True, verbose=True, *args, **kwargs):
        factors_col = self.get_factors(*args, **kwargs)
        if verbose:
            print("We are converting following columns to categorical :{}".format(
                factors_col))
        if inplace:
            self.df.loc[:, factors_col] = self.df.loc[:, factors_col].astype(category)
        else:
            return self.df.loc[:, factors_col].astype(category)

    def remove_category(self, colname, nb_max_levels, replace_value='other', verbose=True):
        """ Replace a variable with too many categories by grouping minor categories to one """
        if self.data.loc[:, colname].nunique() < nb_max_levels:
            if verbose:
                print("{} has not been processed because levels < {}".format(
                    colname, nb_max_levels))
        else:
            if self.is_numeric(colname):
                raise NumericError(
                    '{} is a numeric columns you cannot use this function'.format())
            top_levels = self.data.loc[
                :, colname].value_counts[0:nb_max_levels].index
            self.data.loc[~self.data.loc[:, colname].isin(
                top_levels), colname] = replace_value
