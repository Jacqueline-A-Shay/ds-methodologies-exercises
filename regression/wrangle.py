import pandas as pd
import numpy as np
import env

from env import host, user, password

def wrangle_telco():
    database_name = "telco_churn"
    query = input("Key in the query ")
    url = f'mysql+pymysql://{user}:{password}@{host}/{database_name}'
    reg = pd.read_sql(query,url)
    reg = reg[["customer_id", "tenure", "monthly_charges", "total_charges"]]
    reg.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    reg.total_charges = reg.total_charges.dropna().astype('float')
    return reg
wrangle_telco()