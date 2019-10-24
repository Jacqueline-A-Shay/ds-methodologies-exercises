import pandas as pd
import numpy as np
import acquire
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import acquire
import split_scale

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def process_data(df):
	
	df.drop(columns = 'deck', inplace = True)
	df.fillna(np.nan, inplace = True)

	from sklearn.model_selection import train_test_split
	train, test = train_test_split(df, train_size = 0.8, random_state = 123)

	imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
	
	imp_mode.fit(train[['embarked']])
	train['embarked'] = imp_mode.transform(train[['embarked']])
	test['embarked'] = imp_mode.transform(test[['embarked']])

	imp_mode.fit(train[['embark_town']])
	train['embark_town'] = imp_mode.transform(train[['embark_town']])
	test['embark_town'] = imp_mode.transform(test[['embark_town']])

	imp_mode.fit(train[['age']])
	train['age'] = imp_mode.transform(train[['age']])
	test['age'] = imp_mode.transform(test[['age']])


	LE = LabelEncoder()
	LE.fit(train.embarked)
	train['embarked'] = LE.transform(train[['embarked']])
	test['embarked'] = LE.transform(test[['embarked']])

	scaler, train['age'] = split_scale.my_minmax_scaler(train[['age']])
	scaler, train['fare'] = split_scale.my_minmax_scaler(train[['fare']])

	return df, LE, train, test
