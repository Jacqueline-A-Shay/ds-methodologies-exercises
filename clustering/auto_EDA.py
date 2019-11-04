# distinquish the signals from noises

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
 
import seaborn as sns 
import env
import acquire

# scikit-learn

# Unstructured data 
# no pre-defined data model 
# NoSQL databases and data lakes. 
# e.g. images, video and audio files

# Structured data: 
# pre-defined data models 
# mostly in relational database or data warehouse 
# fixed schema. 
# e.g. transaction information, customers’ information, and dates

# categorical & quantitative

# categorical:

# nominal data & ordinal
# order matters in ordinal

# nominal, e.g. countries 

# ordinal: e.g.
# the competition ranking of a race (1st, 2nd, 3rd)
# the salary grade in an organization (Associate, AVP, VP, SVP)

# quantitative

# interval data (similar to ordinal) 
# > measured along a scale 
# each object’s position is equidistant from one another. 
# thus can perform arithmetic
# e.g. temperature in degrees Fahrenheit 
# where the difference between 78 degrees and 79 degrees 
# same as 45 degrees and 46 degrees

The difference between interval and ratio:
one does not have a true zero point 
while the other does have. 
Example:
When we say something is 0 degrees Fahrenheit, 
it does not mean an absence of heat in that thing. T
his unique property makes statements that involve ratios such as 
“80F twice as hot as 40F” not hold true.


