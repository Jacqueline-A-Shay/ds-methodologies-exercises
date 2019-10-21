df = wrangle.wrangle_telco().set_index("customer_id")
X = df.loc[:, ("tenure", "monthly_charges")]
y = pd.DataFrame(df.total_charges)


# split into train & test
train,test=split_scale.split_my_data(df)
# use standard scaler to scale all data, this step fit and scale
scaler,train_sc,test_sc=split_scale.perform_standard_scaler(train,test)


# split into Xy_unscaled data
X_train = train.drop(columns = 'total_charges')
y_train = train.total_charges
X_test = test.drop(columns = 'total_charges')
y_test = test.total_charges

# split into Xy_scaled data
X_train_sc = train_sc.drop(columns = "total_charges")
y_train_sc = train_sc[["total_charges"]]
X_test_sc = test_sc.drop(columns = "total_charges")
y_test_sc = test_sc[["total_charges"]]

def select_kbest_freg(X_train, y_train, k): 
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression

    f_selector = SelectKBest(f_regression, k).fit(X_train,y_train)
    # f_feature return the selected features
    f_feature = X_train.loc[:,f_support].columns.tolist()
    return f_feature


def ols_backward_elimination(X_train, y_train):
    
    import statsmodels.api as sm

    cols = list(X_train.columns)
    while (len(cols)>0):
        
        X_1 = X_train[cols]
        model = sm.OLS(y_train,X_1).fit()
        p = model.pvalues
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break

    selected_features_BE = cols    
    return selected_features_BE

# penalize the variables not contributing to model
# no removing step
def lasso_cv_coef(X_train, y_train): 
    from sklearn.linear_model import LassoCV
    import matplotlib

    reg = LassoCV().fit(X_train, y_train)
    coef = pd.Series(reg.coef_, index = X_train.columns).sort_values(ascending = False)
    return coef




def find_feature_name(X_train, y_train, number_of_features):
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE
    
    cols = list(X_train.columns)
    model = LinearRegression()

    #Initializing RFE model
    rfe = RFE(model, n)

    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X_train,y_train)  

    #Fitting the data to model
    model.fit(X_rfe,y_train)
    temp = pd.Series(rfe.support_,index = cols)
    selected_features = temp[temp==True].index
    return selected_features

def RFE_feature_name(X_train, y_train, X_test, number_of_features):
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE

    cols = list(X_train.columns)
    model = LinearRegression()
    
    # generate estimator = LinearRegression
    rfe = RFE(model, number_of_features)
    
    train_rfe = rfe.fit_transform(X_train, y_train)
    test_rfe = rfe.transform(X_test)
    
    model.fit(train_rfe, y_train)
    # return True/False for feature
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp == True].index
    
    X_train_rfe = pd.DataFrame(train_rfe, columns = selected_features_rfe)
    X_test_rfe = pd.DataFrame(test_rfe, columns = selected_features_rfe)
    
    return selected_features_rfe, X_train_rfe, X_test_rfe


