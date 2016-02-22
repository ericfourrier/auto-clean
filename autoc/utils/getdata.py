#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: efourrier

Purpose : Get data from https://github.com/ericfourrier/autoc-datasets

"""
import pandas as pd



def get_dataset(name, *args, **kwargs):
    """Get a dataset from the online repo
    https://github.com/ericfourrier/autoc-datasets (requires internet).
    Parameters
    ----------
    name : str
        Name of the dataset 'name.csv'
    """
    path = "https://raw.githubusercontent.com/ericfourrier/autoc-datasets/master/{0}.csv".format(name)
    return pd.read_csv(path, *args, **kwargs)
