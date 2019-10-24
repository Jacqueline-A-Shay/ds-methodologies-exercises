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


def process_col_name():
	df = pd.DataFrame(acquire.get_data_from_mysql())
	df.drop(columns = ["species_id", "measurement_id"], inplace= True)
	df.rename(columns = {"species_name":"species"}, inplace = True)
	print(df.head())
	return df

def split_my_data(data):
	from sklearn.model_selection import train_test_split
	return train_test_split(data, train_size = 0.8, random_state = 123)

def label_encode(train, test):
	from sklearn.preprocessing import LabelEncoder
	LE = LabelEncoder()
	LE.fit(train.species)
	train['species'] = LE.transform(train[['species']])
	test['species'] = LE.transform(test[['species']])
	return LE, train, test

def inv_encode(LE, train, test):
	from sklearn.preprocessing import LabelEncoder
	a = LE.inverse_transform(train[['species']])
	train["species"] = a
	b =  LE.inverse_transform(test[['species']])
	test["species"] = b
	return train, test