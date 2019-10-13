** Evaluate.py 

	- plot_residuals(x, y, df)
	# residual plot, independent variable vs target
        
	- regression_errors(x, y, df)
	# predict yhat w/ linear regression model
	# calc residual for model
	# return SSE, MSE, RMSE, ESS, TSS
	
	- baseline_mean_errors(y,df)
	# calc baseline, aka, yhat = y.mean()
	# return baseline SSE, MSE, RMSE

	- better_than_baseline(x,y)
	# compare model & baseline
	# return SSE, MSE, RMSE for both in a dataframe
	# return difference between model & baseline 
	# return T/F to if model better based on SSE

	- model_significance(x,y,df)
	# return f-statistic/ p-value of f-stat
	# return rsquared
	# return linear model summary in dataframe

	- to be finished:
	# plot linear model/ model formula & baseline/ baseline formula

