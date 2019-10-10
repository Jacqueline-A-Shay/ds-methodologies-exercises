import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import wrangle
import env

from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def obtain_data():
	df = wrangle.wrangle_telco().set_index("customer_id")
	X = df.loc[:, ("tenure", "monthly_charges")]
	y = pd.DataFrame(df.total_charges)
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .80, random_state = 123)
	return (X_train, X_test, y_train, y_test)

# standard
def perform_standard_scaler(X_train, X_test, y_train, y_test):
	X_std_scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
	y_std_scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(y_train)
	
	X_train_scaled = pd.DataFrame(X_std_scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
	X_test_scaled = pd.DataFrame(X_std_scaler.transform(X_test), columns=X_test.columns.values).set_index([X_test.index.values])
	y_train_scaled = pd.DataFrame(y_std_scaler.transform(y_train), columns=y_train.columns.values).set_index([y_train.index.values])
	y_test_scaled = pd.DataFrame(y_std_scaler.transform(y_test), columns=y_test.columns.values).set_index([y_test.index.values])
	return (X_std_scaler, y_std_scaler, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)
def split_scale():
	a, b, c, d = obtain_data()
	return perform_standard_scaler(a, b, c, d)

# uniform
def perform_uniform_scaler(X_train, X_test, y_train, y_test):
    X_u_scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(X_train)
    y_u_scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(y_train)

    X_train_scaled = pd.DataFrame(x_std_scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
    X_test_scaled = pd.DataFrame(x_std_scaler.transform(X_test), columns=X_test.columns.values).set_index([X_test.index.values])

    y_train_scaled = pd.DataFrame(y_std_scaler.transform(y_train), columns=y_train.columns.values).set_index([y_train.index.values])
    y_test_scaled = pd.DataFrame(y_std_scaler.transform(y_test), columns=y_test.columns.values).set_index([y_test.index.values])

# power
# create scaler object using yeo-johnson method and fit to train
X_p_scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(X_train)
y_p_scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(y_train)

X_train_p_scaled = pd.DataFrame(X_p_scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
X_test_p_scaled = pd.DataFrame(X_p_scaler.transform(X_test), columns=X_test.columns.values).set_index([X_test.index.values])

y_train_p_scaled = pd.DataFrame(y_p_scaler.transform(y_train), columns=y_train.columns.values).set_index([y_train.index.values])
y_test_p_scaled = pd.DataFrame(y_p_scaler.transform(y_test), columns=y_test.columns.values).set_index([y_test.index.values])

X_train_p_scaled.head()

# min_max
X_mm_scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X_train)
y_mm_scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(y_train)

X_train_mm_scaled = pd.DataFrame(X_mm_scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
X_test_mm_scaled = pd.DataFrame(X_mm_scaler.transform(X_test), columns=X_test.columns.values).set_index([X_test.index.values])

y_train_mm_scaled = pd.DataFrame(y_mm_scaler.transform(y_train), columns=y_train.columns.values).set_index([y_train.index.values])
y_test_mm_scaled = pd.DataFrame(y_mm_scaler.transform(y_test), columns=y_test.columns.values).set_index([y_test.index.values])

X_train_mm_scaled.head()

# robust
X_r_scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(X_train)
y_r_scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(y_train)

X_train_r_scaled = pd.DataFrame(X_r_scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
X_test_r_scaled = pd.DataFrame(X_r_scaler.transform(X_test), columns=X_test.columns.values).set_index([X_test.index.values])

y_train_r_scaled = pd.DataFrame(y_r_scaler.transform(y_train), columns=y_train.columns.values).set_index([y_train.index.values])
y_test_r_scaled = pd.DataFrame(y_r_scaler.transform(y_test), columns=y_test.columns.values).set_index([y_test.index.values])

