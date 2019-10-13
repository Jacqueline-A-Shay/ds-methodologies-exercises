plot_residuals(x, y, dataframe)

regression_errors(y, yhat)

def regression_errors(x, y, df):
    from statsmodels.formula.api import ols
    # fit linear model
    linear = ols('y~x', data = df).fit()
    # prediction of yhat based on x
    df["yhat"] = linear.predict(pd.DataFrame(x))
    # calc residual =  predict - reality = yhat - y
    df['residual'] = df['yhat'] - y
    # calc SSE, sum(residual^2)
    SSE = sum((df['residual']) ** 2)
    # mean = all result / sample number = SSE/n
    MSE = SSE/ len(df)
    from math import sqrt
    RMSE = sqrt(MSE)
    print("SSE: ", SSE, "MSE: ", MSE, "RMSE: ", RMSE)
    
    return SSE, MSE, RMSE #ESS, TSS
regression_errors(tip.total_bill, tip.tip, tip)