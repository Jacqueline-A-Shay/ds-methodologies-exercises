import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import wrangle
import env

def plot_variable_pairs(df):
	g = sns.pairplot(data = df, kind = 'reg')
	g.map_diag(plt.hist)
	g.map_offdiag(plt.scatter, color='g', alpha = 0.01)
	return g

def months_to_years(df):
	df["tenure_years"] = round(df.tenure / 12).astype('category')
	tenure_in_year = df
	return tenure_in_year

def plot_categorical_and_continous_vars(tenure_in_year):
	fig = plt.figure(figsize=(25, 25))
	gs = gridspec.GridSpec(4, 3)
	sns.set(font_scale=1.4)
	ax1 = plt.subplot(gs[0, 0])
	ax2 = plt.subplot(gs[0, 1])
	ax3 = plt.subplot(gs[0, -1])
	ax4 = plt.subplot(gs[1: , :])

	sns.violinplot(x="tenure_years", y="total_charges", data=tenure_in_year, ax = ax1)
	sns.boxplot(x="tenure_years", y="total_charges", data=tenure_in_year, ax = ax2)
	sns.barplot(x="tenure_years", y="total_charges", data=tenure_in_year, ax = ax3)
	sns.heatmap(tenure_in_year.corr(),cmap='Blues',annot=True, ax = ax4)

def final_plot():
	
	df = wrangle.wrangle_telco().set_index("customer_id")
	plot_pairs = plot_variable_pairs(df)
	tenure_in_year = months_to_years(df)
	plot_category = plot_categorical_and_continous_vars(tenure_in_year)

	return plot_pairs, plot_category