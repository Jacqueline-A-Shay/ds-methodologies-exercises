import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import wrangle
import env

from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

df = wrangle.wrangle_telco().set_index("customer_id")
X = df.loc[:, ("tenure", "monthly_charges")]
y = pd.DataFrame(df.total_charges)

# split dataframe into train(train_percent: 80%) & test(20%)
def split_my_data(df):
	train, test = train_test_split(df, train_size = 0.8, random_state = 123)
	return train, test
# split_my_data(df)

# standard
def perform_standard_scaler(train, test):
	scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
	
	train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
	test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
	return scaler, train_scaled, test_scaled


# def split_scale():
# 	a, b, c, d = obtain_data()
# 	return perform_standard_scaler(a, b, c, d)

# uniform
def perform_uniform_scaler(train, test):
	u_scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(train)
	u_train_scaled = pd.DataFrame(u_scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
	u_test_scaled = pd.DataFrame(u_scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
	return u_scaler, u_train_scaled, u_test_scaled

# power
# create scaler object using yeo-johnson method and fit to train
def gaussian_scaler(train, test):
	p_scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(train)
	p_train_scaled = pd.DataFrame(p_scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
	p_test_scaled = pd.DataFrame(p_scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
	return p_scaler, p_train_scaled, p_test_scaled

# min_max
def min_max_scaler(train, test):
	mm_scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)
	mm_train_scaled = pd.DataFrame(mm_scaler.transform(train), columns= train.columns.values).set_index([train.index.values])
	mm_test_scaled = pd.DataFrame(mm_scaler.transform(test), columns= test.columns.values).set_index([test.index.values])
	return mm_scaler, mm_train_scaled, mm_test_scaled


# robust
def iqr_robust_scaler(train, test):
	r_scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)

	r_train_scaled = pd.DataFrame(r_scaler.transform(train), columns= train.columns.values).set_index([train.index.values])
	r_test_scaled = pd.DataFrame(r_scaler.transform(test), columns= test.columns.values).set_index([test.index.values])
	return r_scaler, r_train_scaled, r_test_scaled
