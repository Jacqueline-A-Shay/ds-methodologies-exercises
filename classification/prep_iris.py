import pandas as pd
import numpy as np
import acquire
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


	# query = '''
	# SELECT * FROM measurements JOIN species USING(species_id);
	# '''

def process_col_name(df):
	
	#df = pd.DataFrame(acquire.get_data_from_mysql())
	df.drop(columns = ["species_id", "measurement_id"], inplace= True)
	df.rename(columns = {"species_name":"species"}, inplace = True)
	print(df.head())
	return df

def split_my_data(data):
	from sklearn.model_selection import train_test_split
	return train_test_split(data, train_size = 0.7, random_state = 123)

