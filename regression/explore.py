import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import wrangle
import env


def plot_variable_pairs(df):
# 	df = wrangle.wrangle_telco().set_index("customer_id")
 	g = sns.pairplot(df, kind = 'reg')
 	g.map_diag(plt.hist)
 	g.map_offdiag(plt.scatter)
 	return g

def months_to_years(df):
	df["tenure_years"] = round(df.tenure / 12).astype('category')
	tenure_in_year = df
	return tenure_in_year

def plot_categorical_and_continous_vars(tenure_in_year):
	f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
	sns.violinplot(x="tenure_years", y="total_charges", data=tenure_in_year, ax = ax1)
	sns.boxplot(x="tenure_years", y="total_charges", data=tenure_in_year, ax = ax2)
	sns.barplot(x="tenure_years", y="total_charges", data=tenure_in_year, ax = ax3)

def final_plot(df):
	return plot_categorical_and_continous_vars(months_to_years(df))