import pandas as pd
import numpy as np
import env

def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_data_from_mysql():
    query = input("Type in query")
    db = input("Name the database")
    df = pd.read_sql(query, get_db_url(db))
    return df

