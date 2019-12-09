import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib default plotting styles
plt.rc("patch", edgecolor="black", force_edgecolor=True)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle=":", linewidth=0.8, alpha=0.7)
plt.rc("axes.spines", right=False, top=False)
plt.rc("figure", figsize=(11, 8))
plt.rc("font", size=12.0)
plt.rc("hist", bins=25)

import acquire


# From the Series we can extract the value_counts, which is our raw count
# for term frequency. Once we have the raw counts, we can calculate the
# other measures.
def freq(df):
	"""
	calculate raw count, frequency, augmented frequency
	- Raw Count: count of the number of occurances of each word
	- Frequency: The number of times each word appears divided by the total number of words.
	- Augmented Frequency: The frequency of each word divided by the maximum frequency. 
		This can help prevent bias towards larger documents.

	"""
	freq_calc = (pd.DataFrame({'raw_count': words.value_counts()})
			.assign(frequency=lambda df: df.raw_count / df.raw_count.sum())
			.assign(augmented_frequency=lambda df: df.frequency / df.frequency.max()))
	return freq_calc