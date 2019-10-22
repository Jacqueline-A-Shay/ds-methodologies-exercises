import pandas as pd
import numpy as np
import env
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_data_from_mysql():
    query = input("Type in your query")
    db = input("Name the database")
    df = pd.read_sql(query, get_db_url(db))
    return df



# def find_outliers(data, column):
#     Q1 = data[column].quantile(0.25)
#     Q3 = data[column].quantile(0.75)
#     IQR = Q3 - Q1
#     max_value = Q3 * 1.5
#     min_value = Q1 * 1.5
    
#     if data[column].max() < max_value and data[column].min() > min_value:
#         print("No outliers found")
#     else:
#         print("OUTLIER DETECTED!!!")
#     f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=[15, 7])
    
#     f.subplots_adjust(hspace=.4)
    
#     sns.boxplot(data[column].dropna(), ax=ax0, color="#34495e").set_title('Before')
#     sns.distplot(data[column].dropna(), ax=ax2, color="#34495e").set_title('Before')

#     data.loc[data[column] > max_value, column] = max_value
#     data.loc[data[column] < min_value, column] = min_value
    
#     sns.boxplot(data[column].dropna(), ax=ax1, color="#34495e").set_title('After')
#     sns.distplot(data[column].dropna(), ax=ax3, color="#34495e").set_title('After')
  


def clean_data(df):
    df = df.drop((missing_values.Column_Name[missing_values.Percent_Missing >= 0.7]), axis = 1).set_index("parcelid")
    # Disern Categorical vs Numerical Data
    numerical = df[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',\
                     'garagecarcnt', 'garagetotalsqft','roomcnt', 'unitcnt', 'yearbuilt','lotsizesquarefeet']]

    categorical = df[['buildingqualitytypeid', 'fips', 'heatingorsystemtypeid'\
                      'propertylandusetypeid', 'propertyzoningdesc', 'rawcensustractandblock',\
                       'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip']]

    tax_related = df[['fips','structuretaxvaluedollarcnt','taxvaluedollarcnt', 'assessmentyear',\
                       'landtaxvaluedollarcnt','taxamount', 'censustractandblock']]

    geo = df[['taxvaluedollarcnt','fips','latitude', 'longitude']]

    values = {"garagecarcnt": numerical.garagecarcnt.median(),\
          "unitcnt": numerical.unitcnt.median(),\
          "garagetotalsqft":numerical.garagetotalsqft.median(),\
          "lotsizesquarefeet":numerical.lotsizesquarefeet.median(),\
             "yearbuilt":numerical.yearbuilt.median()}
    numerical = numerical.fillna(value=values)
    
    return df


def wrangle():
    df = get_data_from_mysql()
    df = clean_data(df)
    return df
