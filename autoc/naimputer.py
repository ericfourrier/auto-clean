from autoc.explorer import DataExploration




class NaImputer(object):

    def __init__(self, *args, *kwargs):
        super(DataExploration,self).__init__(*args, *kwargs)

	@staticmethod
	def fillna_serie(serie,special_value = None ):
		""" fill values in a serie default with the mean for numeric or the most common
		factor for categorical variable """

		if special_value:
			return serie.fillna(serie.mean())
		if (serie.dtype ==  float) | (serie.dtype == int) :
			return serie.fillna(serie.median())
		else:
            if special_value is not None:
                serie.fillna(special_value) # "Missing for example"
			return serie.fillna(serie.value_counts().index[0])

	def basic_naimputation(self,columns_to_process = [],threshold = None):
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
			columns_to_process = columns_to_process + cserie(self.nacolcount().Napercentage < threshold)
		df.loc[:,columns_to_process] = df.loc[:,columns_to_process].apply(lambda x: self.fillna_serie(x))
		return df
