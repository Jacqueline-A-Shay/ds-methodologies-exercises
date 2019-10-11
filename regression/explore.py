import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import wrangle
import env


# def plot_variable_pairs():
# 	df = wrangle.wrangle_telco().set_index("customer_id")
# 	g = sns.pairplot(df, kind = 'reg')
# 	g.map_diag(plt.hist)
# 	g.map_offdiag(plt.scatter)
# 	return g

g = sns.PairGrid(train)# create grids
g.map_diag(plt.hist) # fill in histogram in diagnal
g.map_offdiag(plt.scatter) # fill in remaining with scatter



def months_to_years():
	df = wrangle.wrangle_telco().set_index("customer_id")
	df["tenure_years"] = round(df.tenure / 12).astype('category')
	return df

def plot_categorical_and_continous_vars():
	v = sns.violinplot(x="tenure_years", y="total_charges", data=df)
	b = sns.boxplot(x="tenure_years", y="total_charges", data=df)
	b = sns.barplot(x="tenure_years", y="total_charges", data=df)
