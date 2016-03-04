# -*- coding: utf-8 -*-
"""

@author: efourrier

Purpose : Automated test suites with unittest
run "python -m unittest -v test" in the module directory to run the tests

The clock decorator in utils will measure the run time of the test
"""

#########################################################
# Import Packages and helpers
#########################################################

import unittest
# internal helpers
# from autoc.utils.helpers import clock, create_test_df, removena_numpy, cserie
from autoc.utils.helpers import random_pmf, clock, create_test_df, cserie, simu, removena_numpy
from autoc.utils.getdata import get_dataset
from autoc.explorer import DataExploration
from autoc.naimputer import NaImputer
import pandas as pd
import numpy as np


flatten_list = lambda x: [y for l in x for y in flatten_list(
    l)] if isinstance(x, list) else [x]


# flatten_list = lambda x: [y for l in x for y in flatten_list(l)] if isinstance(x,list) else [x]
#########################################################
# Writing the tests
#########################################################


class TestDataExploration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ creating test data set for the test module """
        cls._test_df = create_test_df()
        cls._test_dc = DataExploration(data=cls._test_df)

    @clock
    def test_inplace(self):
        exploration_copy = DataExploration(data=create_test_df(), inplace=True)
        self.assertEqual(id(self._test_df), id(self._test_dc.data))
        self.assertNotEqual(id(self._test_df), id(exploration_copy.data))

    @clock
    def test_cserie(self):
        char_var = cserie(self._test_dc.data.dtypes == "object")
        self.assertIsInstance(char_var, list)
        self.assertIn('character_variable', char_var)

    @clock
    def test_removena_numpy(self):
        test_array = np.array([np.nan, 1, 2, np.nan])
        self.assertTrue((removena_numpy(test_array) == np.array([1, 2])).all())

    @clock
    def test_sample_df(self):
        self.assertEqual(len(self._test_dc.sample_df(pct=0.061)),
                         0.061 * float(self._test_dc.data.shape[0]))

    @clock
    def test_nrow(self):
        self.assertEqual(self._test_dc._nrow, self._test_dc.data.shape[0])

    @clock
    def test_col(self):
        self.assertEqual(self._test_dc._ncol, self._test_dc.data.shape[1])

    @clock
    def test_is_numeric(self):
        self.assertTrue(self._test_dc.is_numeric("num_variable"))
        self.assertTrue(self._test_dc.is_numeric("many_missing_70"))
        self.assertFalse(self._test_dc.is_numeric("character_variable"))

    @clock
    def test_is_int_factor(self):
        self.assertFalse(self._test_dc.is_int_factor("num_variable"))
        self.assertTrue(self._test_dc.is_int_factor("int_factor_10", 0.01))
        self.assertTrue(self._test_dc.is_int_factor("int_factor_10", 0.1))
        self.assertFalse(self._test_dc.is_int_factor("int_factor_10", 0.005))
        self.assertFalse(self._test_dc.is_int_factor("character_variable"))

    @clock
    def test_where_numeric(self):
        self.assertEqual(cserie(self._test_dc.where_numeric().all()), self._test_dc._dfnum)


    @clock
    def test_total_missing(self):
        self.assertEqual(self._test_dc.total_missing,
                         self._test_dc.data.isnull().sum().sum())

    @clock
    def test_nacolcount_capture_na(self):
        nacolcount = self._test_dc.nacolcount()
        self.assertEqual(nacolcount.loc['na_col', 'Napercentage'], 1.0)
        self.assertEqual(
            nacolcount.loc['many_missing_70', 'Napercentage'], 0.7)

    @clock
    def test_nacolcount_is_type_dataframe(self):
        self.assertIsInstance(self._test_dc.nacolcount(),
                              pd.core.frame.DataFrame)

    @clock
    def test_narowcount_capture_na(self):
        narowcount = self._test_dc.narowcount()
        self.assertEqual(sum(narowcount['Nanumber'] > 0), self._test_dc._nrow)

    @clock
    def test_narowcount_is_type_dataframe(self):
        narowcount = self._test_dc.narowcount()
        self.assertIsInstance(narowcount, pd.core.frame.DataFrame)

    @clock
    def test_manymissing_capture(self):
        manymissing = self._test_dc.manymissing(0.7)
        self.assertIsInstance(manymissing, list)
        self.assertIn('many_missing_70', manymissing)
        self.assertIn('na_col', manymissing)

    @clock
    def test_nacols_full(self):
        nacols_full = self._test_dc.nacols_full
        self.assertIsInstance(nacols_full, list)
        self.assertIn('na_col',nacols_full )

    @clock
    def test_narows_full(self):
        test_df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
        test_df.loc[99, :] = np.nan
        self.assertIn(99, DataExploration(test_df).narows_full)
        self.assertNotIn(1, test_df)

    @clock
    def test_constant_col_capture(self):
        constantcol = self._test_dc.constantcol()
        self.assertIsInstance(constantcol, list)
        self.assertIn('constant_col', constantcol)
        self.assertIn('constant_col_num', constantcol)
        self.assertIn('na_col', constantcol)

    @clock
    def test_count_unique(self):
        count_unique = self._test_dc.count_unique()
        self.assertIsInstance(count_unique, pd.Series)
        self.assertEqual(count_unique.id, 1000)
        self.assertEqual(count_unique.constant_col, 1)
        self.assertEqual(count_unique.character_factor, 7)

    @clock
    def test_dfchar_check_col(self):
        dfchar = self._test_dc._dfchar
        self.assertIsInstance(dfchar, list)
        self.assertNotIn('num_variable', dfchar)
        self.assertIn('character_factor', dfchar)
        self.assertIn('character_variable', dfchar)
        self.assertNotIn('many_missing_70', dfchar)

    @clock
    def test_dfnum_check_col(self):
        dfnum = self._test_dc._dfnum
        self.assertIsInstance(dfnum, list)
        self.assertIn('num_variable', dfnum)
        self.assertNotIn('character_factor', dfnum)
        self.assertNotIn('character_variable', dfnum)
        self.assertIn('many_missing_70', dfnum)

    @clock
    def test_factors_check_col(self):
        factors = self._test_dc.factors()
        self.assertIsInstance(factors, list)
        self.assertNotIn('num_factor', factors)
        self.assertNotIn('character_variable', factors)
        self.assertIn('character_factor', factors)

    @clock
    def test_detectkey_check_col(self):
        detectkey = self._test_dc.detectkey()
        self.assertIsInstance(detectkey, list)
        self.assertIn('id', detectkey)
        self.assertIn('member_id', detectkey)

    @clock
    def test_detectkey_check_col_dropna(self):
        detectkeyna = self._test_dc.detectkey(dropna=True)
        self.assertIn('id_na', detectkeyna)
        self.assertIn('id', detectkeyna)
        self.assertIn('member_id', detectkeyna)

    @clock
    def test_findupcol_check(self):
        findupcol = self._test_dc.findupcol()
        self.assertIn(['id', 'duplicated_column'], findupcol)
        self.assertNotIn('member_id', flatten_list(findupcol))

    @clock
    def test_count_unique(self):
        count_unique = self._test_dc.count_unique()
        self.assertIsInstance(count_unique, pd.Series)
        self.assertEqual(count_unique.id, len(self._test_dc.data.id))
        self.assertEqual(count_unique.constant_col, 1)
        self.assertEqual(count_unique.num_factor, len(
            pd.unique(self._test_dc.data.num_factor)))

    @clock
    def test_structure(self):
        structure = self._test_dc.structure()
        self.assertIsInstance(structure, pd.DataFrame)
        self.assertEqual(len(self._test_dc.data),
                         structure.loc['na_col', 'nb_missing'])
        self.assertEqual(len(self._test_dc.data), structure.loc[
                         'id', 'nb_unique_values'])
        self.assertTrue(structure.loc['id', 'is_key'])

    @clock
    def test_nearzerovar(self):
        nearzerovar = self._test_dc.nearzerovar(save_metrics=True)
        self.assertIsInstance(nearzerovar, pd.DataFrame)
        self.assertIn('nearzerovar_variable', cserie(nearzerovar.nzv))
        self.assertIn('constant_col', cserie(nearzerovar.nzv))
        self.assertIn('na_col', cserie(nearzerovar.nzv))


class TestNaImputer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ creating test data set for the test module """
        cls._test_na = NaImputer(data=create_test_df())

    @clock
    def test_fillna_serie(self):
        test_serie = pd.Series([1, 3, np.nan, 5])
        self.assertIsInstance(
            self._test_na.fillna_serie(test_serie), pd.Series)
        self.assertEqual(self._test_na.fillna_serie(test_serie)[2], 3.0)

    @clock
    def test_fillna_serie(self):
        test_char_variable = self._test_na.fillna_serie('character_variable_fillna')
        test_num_variable = self._test_na.fillna_serie('numeric_variable_fillna')
        self.assertTrue(test_char_variable.notnull().any())
        self.assertTrue(test_num_variable.notnull().any())
        self.assertTrue((pd.Series(
            ['A'] * 300 + ['B'] * 200 + ['C'] * 200 + ['A'] * 300) == test_char_variable).all())
        self.assertTrue(
            (pd.Series([1] * 400 + [3] * 400 + [2] * 200) == test_num_variable).all())

    @clock
    def test_fill_low_na(self):
        df_fill_low_na = self._test_na.basic_naimputation(columns_to_process=['character_variable_fillna',
                                                                       'numeric_variable_fillna'])
        df_fill_low_na_threshold = self._test_na.basic_naimputation(threshold=0.4)
        self.assertIsInstance(df_fill_low_na, pd.DataFrame)
        self.assertIsInstance(df_fill_low_na_threshold, pd.DataFrame)
        self.assertTrue((pd.Series(['A'] * 300 + ['B'] * 200 + ['C'] * 200 + [
                        'A'] * 300) == df_fill_low_na.character_variable_fillna).all())
        self.assertTrue((pd.Series([1] * 400 + [3] * 400 + [2] * 200)
                         == df_fill_low_na.numeric_variable_fillna).all())
        self.assertTrue((pd.Series(['A'] * 300 + ['B'] * 200 + ['C'] * 200 + [
                        'A'] * 300) == df_fill_low_na_threshold.character_variable_fillna).all())
        self.assertTrue((pd.Series([1] * 400 + [3] * 400 + [2] * 200)
                         == df_fill_low_na_threshold.numeric_variable_fillna).all())
        self.assertTrue(
            sum(pd.isnull(df_fill_low_na_threshold.many_missing_70)) == 700)


class TestHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ creating test data set for the test module """
        cls.data = create_test_df()

    @clock
    def test_random_pmf(self):
        self.assertAlmostEqual(len(random_pmf(10)), 10)
        self.assertAlmostEqual(random_pmf(10).sum(), 1)
    @clock
    def test_simu(self):
        pmf = random_pmf(4)
        samples_unique = simu((np.array(['A', 'B']), np.array([0, 1])), 10)
        self.assertTrue((samples_unique == 'B').all())


class TestGetData(unittest.TestCase):

    @clock
    def test_getdata_titanic(self):
        """ Test if downloading titanic data is working """
        titanic = get_dataset('titanic')
        self.assertIsInstance(titanic, pd.DataFrame)
        self.assertEqual(titanic.shape[0], 891)
        self.assertEqual(titanic.shape[1], 15)



# Adding new tests sets
# def suite():
#    suite = unittest.TestSuite()
#    suite.addTest(TestPandasPatch('test_default_size'))
#    return suite
# Other solution than calling main

#suite = unittest.TestLoader().loadTestsFromTestCase(TestPandasPatch)
#unittest.TextTestRunner(verbosity = 1 ).run(suite)

if __name__ == "__main__":
    unittest.main(exit=False)
