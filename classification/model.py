import numpy as np
import pandas as pd
import seaborn as sns

from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def linear_model(X_train, y_train):
    df = pd.DataFrame(y_train)
    lm=LinearRegression()
    lm.fit(X_train,y_train)
    lm_predictions=lm.predict(X_train)
    df['lm']=lm_predictions
    return df

def compute_baseline(y):
    return np.array([y.mean()]*len(y))

def evaluate(actual, model):
    MSE = mean_squared_error(actual, model)
    SSE = MSE*len(actual)
    RMSE = sqrt(MSE)
    r2 = r2_score(actual, model)
    print("MSE: ", MSE, "SSE: ", SSE, "RMSE: ",RMSE, "r^2: ", r2)
    return MSE, SSE, RMSE, r2 
    
def plot_linear_model(actuals, lm, baseline):
    plot = pd.DataFrame({'actual': actuals,
                'lm': lm,
                'baseline': baseline.flatten()})\
    .melt(id_vars=['actual'], var_name='model', value_name='prediction')\
    .pipe((sns.relplot, 'data'), x='actual', y='prediction', hue='model')
    return plot

def create_polynomial_regression_model_1(degree, y_train, X_train_feature, X_test_feature, y_train_feature, y_test_feature):
    result = pd.DataFrame(y_train)
    result['actual'] = pd.DataFrame(y_train)
    #result['test_actual'] = pd.DataFrame(y_test)
    result['baseline'] = np.array([y_train.mean()]*len(y_train))
    
    "Creates a polynomial regression model for the given degree"
    poly_features = PolynomialFeatures(degree=degree)
    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train_feature)
    # fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train_feature)

    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
    result['y_train_pred']=y_train_predicted
    
    # predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test_feature))
    result['y_test_pred']=y_train_predicted
    
    # evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(y_train_feature, y_train_predicted))
    r2_train = r2_score(y_train_feature, y_train_predicted)
  
    # evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(y_test_feature, y_test_predict))
    r2_test = r2_score(y_test_feature, y_test_predict)
    
    print("The model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))
  
    print("\n")
  
    print("The model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))

def create_polynomial_regression_model_2(degree, y_train, X_train_ff, X_test_ff, y_train_ff, y_test_ff):
    result2 = pd.DataFrame(y_train)
    result2['actual'] = pd.DataFrame(y_train_ff)
    #result['test_actual'] = pd.DataFrame(y_test)
    result2['baseline'] = np.array([y_train_ff.mean()]*len(y_train_ff))
    
    "Creates a polynomial regression model for the given degree"
    poly_features = PolynomialFeatures(degree=degree)
    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train_ff)
    # fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train_ff)

    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
    result2['y_train_pred']=y_train_predicted
    
    # predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test_ff))
    result2['y_test_pred']=y_train_predicted
    
    # evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(y_train_ff, y_train_predicted))
    r2_train = r2_score(y_train_feature, y_train_predicted)
  
    # evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(y_test_ff, y_test_predict))
    r2_test = r2_score(y_test_feature, y_test_predict)
    
    print("The model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))
  
    print("\n")
  
    print("The model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))
    return result2