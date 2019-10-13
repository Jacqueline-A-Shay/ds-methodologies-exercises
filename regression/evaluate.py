plot_residuals(x, y, dataframe)


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
    ESS = sum((df['yhat'] - y.mean())**2)
    TSS = ESS + SSE
    print("SSE: ", SSE, "MSE: ", MSE, "RMSE: ", RMSE)
    return SSE, MSE, RMSE, ESS, TSS
regression_errors(tip.total_bill, tip.tip, tip)


def baseline_mean_errors(y,df):
    # compare w/ baseline, if not better, no need model, just rely on mean or median

    # copy dataframe
    df_baseline = pd.DataFrame(y)
    # compute the overall mean of the y values and add to 'yhat' as our prediction
    df_baseline['yhat'] = df_baseline.tip.mean()
    
    # SSE_baseline
    # compute the difference between y and yhat
    df_baseline['residual'] = df_baseline['yhat'] - df_baseline.tip
    # calc SSE_baseline from residual
    SSE_baseline = sum(df_baseline.residual ** 2)
    # MSE_baseline
    MSE_baseline = SSE_baseline/len(df_baseline)
    # RMSE_baseline
    RMSE_baseline = sqrt(MSE_baseline)
    
    print("SSE_base: ", SSE_baseline, "MSE_base: ", MSE_baseline, "RMSE_base: ", RMSE_baseline)
    return SSE_baseline, MSE_baseline, RMSE_baseline
baseline_mean_errors(tip.tip, tip)

def better_than_baseline(x,y):

    # create dataframe holding summary
    df_eval = pd.DataFrame(np.array(['SSE','MSE','RMSE']), columns=['metric'])
    df_eval['model_error'] = np.array([x[0], x[1], x[2]])

    df_eval['baseline_error'] = np.array([y[0], y[1], y[2]])
    df_eval['error_delta'] = df_eval.model_error - df_eval.baseline_error
    print(df_eval)
    if x[0] < y[0]:
        return True
    else:
        False
better_than_baseline(regression_errors(tip.total_bill, tip.tip, tip),baseline_mean_errors(tip.tip, tip))



# def regression_errors(x, y, df):
#     from statsmodels.formula.api import ols
#     # fit linear model
#     linear = ols('y~x', data = df).fit()
#     # prediction of yhat based on x
#     df["yhat"] = linear.predict(pd.DataFrame(x))
#     # calc residual =  predict - reality = yhat - y
#     df['residual'] = df['yhat'] - y
#     # calc SSE, sum(residual^2)
#     SSE = sum((df['residual']) ** 2)
#     # mean = all result / sample number = SSE/n
#     MSE = SSE/ len(df)
#     from math import sqrt
#     RMSE = sqrt(MSE)
    
#     # compare w/ baseline, if not better, no need model, just rely on mean or median

#     # copy dataframe
#     df_baseline = pd.DataFrame(x)
#     df_baseline['target'] = y
#     # compute the overall mean of the y values and add to 'yhat' as our prediction
#     df_baseline['yhat'] = df_baseline.target.mean()
    
#     # SSE_baseline
#     # compute the difference between y and yhat
#     df_baseline['residual'] = df_baseline['yhat'] - df_baseline.target
#     # calc SSE_baseline from residual
#     SSE_baseline = sum(df_baseline.residual ** 2)
#     # MSE_baseline
#     MSE_baseline = SSE_baseline/len(df_baseline)
#     # RMSE_baseline
#     RMSE_baseline = sqrt(MSE_baseline)

#     # create dataframe holding summary
#     df_eval = pd.DataFrame(np.array(['SSE','MSE','RMSE']), columns=['metric'])
#     df_eval['model_error'] = np.array([SSE, MSE, RMSE])

#     df_eval['baseline_error'] = np.array([SSE_baseline, MSE_baseline, RMSE_baseline])
#     df_eval['error_delta'] = df_eval.model_error - df_eval.baseline_error
   
#     print("SSE: ", SSE, "MSE: ", MSE, "RMSE: ", RMSE)
#     print("SSE_base: ", SSE_baseline, "MSE_base: ", MSE_baseline, "RMSE_base: ", RMSE_baseline)
    
#     return df_eval, SSE, MSE, RMSE #ESS, TSS
#regression_errors(tip.total_bill, tip.tip, tip)
