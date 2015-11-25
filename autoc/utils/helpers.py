# -*- coding: utf-8 -*-
"""
@author: efourrier

Purpose : Create toolbox functions to use for the different pieces of code ot the package

"""
from numpy.random import normal
from numpy.random import choice
import time
import pandas as pd
import numpy as np


def cserie(serie):
    return serie[serie].index.tolist()


def removena_numpy(array):
    return array[~(np.isnan(array))]


def common_cols(df1, df2):
    """ Return the intersection of commun columns name """
    return list(set(df1.columns) & set(df2.columns))


def bootstrap_ci(x, n=300, ci=0.95):
    """
    this is a function depending on numpy to compute bootstrap percentile
    confidence intervalfor the mean of a numpy array

    Arguments
    ---------
    x : a numpy ndarray
    n : the number of boostrap samples
    ci : the percentage confidence (float) interval in ]0,1[

    Return
    -------
    a tuple (ci_inf,ci_up)
    """

    low_per = 100 * (1 - ci) / 2
    high_per = 100 * ci + low_per
    x = removena_numpy(x)
    if not len(x):
        return (np.nan, np.nan)
    bootstrap_samples = choice(a=x, size=(
        len(x), n), replace = True).mean(axis = 0)
    return np.percentile(bootstrap_samples, [low_per, high_per])


def clock(func):
    """ decorator to measure the duration of each test of the unittest suite,
    this is extensible for any kind of functions it will just add a print  """
    def clocked(*args):
        t0 = time.time()
        result = func(*args)
        elapsed = (time.time() - t0) * 1000  # in ms
        print('elapsed : [{0:0.3f}ms]'.format(elapsed))
        return result
    return clocked


def create_test_df():
    """ Creating a test pandas DataFrame for the unittest suite """
    test_df = pd.DataFrame({'id': [i for i in range(1, 1001)], 'member_id': [
                           10 * i for i in range(1, 1001)]})
    test_df['na_col'] = np.nan
    test_df['id_na'] = test_df.id
    test_df.loc[1:3, 'id_na'] = np.nan
    test_df['constant_col'] = 'constant'
    test_df['constant_col_num'] = 0
    test_df['character_factor'] = [
        choice(list('ABCDEFG')) for _ in range(1000)]
    test_df['num_factor'] = [choice([1, 2, 3, 4]) for _ in range(1000)]
    test_df['nearzerovar_variable'] = 'most_common_value'
    test_df.loc[0, 'nearzerovar_variable'] = 'one_value'
    test_df['binary_variable'] = [choice([0, 1]) for _ in range(1000)]
    test_df['character_variable'] = [str(i) for i in range(1000)]
    test_df['duplicated_column'] = test_df.id
    test_df['many_missing_70'] = [1] * 300 + [np.nan] * 700
    test_df['character_variable_fillna'] = ['A'] * \
        300 + ['B'] * 200 + ['C'] * 200 + [np.nan] * 300
    test_df['numeric_variable_fillna'] = [1] * 400 + [3] * 400 + [np.nan] * 200
    test_df['num_variable'] = 100
    test_df['outlier'] = normal(size=1000)
    test_df.loc[[1, 10, 100], 'outlier'] = [10, 5, 10]
    return test_df


def simu(pmf, size):
    """ Draw one sample from of a discrete distribution, pmf is supposed to
    be in ascending order

    Parameters
    ----------
    pmf : tuple(ndarray, ndarray)
        a tuple with (labels,probs) labels are supposed to be in ascending order
    size: int
        the number of sampel you want generate
    Return
    ------
    int (depends of the type of labels)
        draw a random sample from the pmf
    """
    labels, probs = pmf[0], pmf[1]
    u = np.random.rand(size)
    cumulative_sum = probs.cumsum()
    return labels[(u >= cumulative_sum[:, None]).argmin(axis=0)]


def random_pmf(nb_labels):
    """ Return a random probability mass function of nb_labels"""
    random_numbers = np.random.random(nb_labels)
    return random_numbers / np.sum(random_numbers)


def random_histogram(nb_labels, nb_observations):
    """ Return a random probability mass function of nb_labels"""
    random_histo = np.random.choice(np.arange(0, nb_observations), nb_labels)
    return random_histo / np.sum(random_histo)


def simulate_na_col(df, colname, pct, weights=None, random_weights=True):
    """ Simulate missing values in a column of categorical variables """
    # if df.loc[:,colname].dtype == 'float' or df.loc[:,colname].dtype == 'int':
    #     raise TypeError('This function only support categorical variables')
    pmf = df.loc[:, column].value_counts(normalize=True)
    labels = pmf.index.values  # characters
    pmf_na = weights if weights else random_pmf(len(labels))
    na_simu = simu((labels, pmf.values), int(pct*len(df.index)))
    return na_simu


def get_test_df_complete():
    """ get the full test dataset from Lending Club open source database,
    the purpose of this fuction is to be used in a demo ipython notebook """
    import requests
    from zipfile import ZipFile
    from io import StringIO
    zip_to_download = "https://resources.lendingclub.com / LoanStats3b.csv.zip"
    r = requests.get(zip_to_download)
    zipfile = ZipFile(StringIO(r.content))
    file_csv = zipfile.namelist()[0]
    # we are using the c parser for speed
    df = pd.read_csv(zipfile.open(file_csv), skiprows=[0], na_values=['n/a', 'N/A', ''],
                     parse_dates=['issue_d', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d'])
    zipfile.close()
    df = df[:-2]
    nb_row = float(len(df.index))
    df['na_col'] = np.nan
    df['constant_col'] = 'constant'
    df['duplicated_column'] = df.id
    df['many_missing_70'] = np.nan
    df.loc[1:int(0.3 * nb_row), 'many_missing_70'] = 1
    df['bad'] = 1
    index_good = df['loan_status'].isin(
        ['Fully Paid', 'Current', 'In Grace Period'])
    df.loc[index_good, 'bad'] = 0
    return df


def psi(bench, target, group, print_df=True):
    """ This function return the Population Stability Index, quantifying if the
    distribution is stable between two states.
    This statistic make sense and works is only working for numeric variables
    for bench and target.
    Params:
    - bench is a numpy array with the reference variable.
    - target is a numpy array of the new variable.
    - group is the number of group you want consider.
    """
    labels_q = np.percentile(
        bench, [(100.0 / group) * i for i in range(group + 1)], interpolation="nearest")

    # This is the right approach when you have not a lot of unique value
    ben_pct = (pd.cut(bench, bins=np.unique(labels_q),
                      include_lowest=True).value_counts()) / len(bench)
    target_pct = (pd.cut(target, bins=np.unique(labels_q),
                         include_lowest=True).value_counts()) / len(target)
    target_pct = target_pct.sort_index()  # sort the index
    ben_pct = ben_pct.sort_index()  # sort the index
    psi = sum((target_pct - ben_pct) * np.log(target_pct / ben_pct))
    # Print results for better understanding
    if print_df:
        results = pd.DataFrame({'ben_pct': ben_pct.values,
                                'target_pct': target_pct.values},
                               index=ben_pct.index)
        return {'data': results, 'statistic': psi}
    return psi
