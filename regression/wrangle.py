import pandas as pd
import numpy as np

import env

# def get_db_url(db):
#     return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

# def get_data_from_mysql():
#     query = '''
#     SELECT *
#     FROM customers
#     JOIN internet_service_types USING (internet_service_type_id)
#     WHERE contract_type_id = 3
#     '''

#     df = pd.read_sql(query, get_db_url('telco_churn'))
#     return df

# def clean_data(df):
#     df = df[['customer_id', 'total_charges', 'monthly_charges', 'tenure']]
#     df.total_charges = df.total_charges.str.strip().replace('', np.nan).astype(float)
#     df = df.dropna()
#     return df
  
# def wrangle_telco():
#     df = get_data_from_mysql()
#     df = clean_data(df)
#     return df
    
# # def wrangle_telco():
# #     return clean_data(get_data_from_mysql())






def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def wrangle_telco():
    query = '''
    select * from customers  
    join contract_types USING (contract_type_id)
    where contract_type_id = 3
    '''
    reg = pd.read_sql(query,get_db_url('telco_churn'))
    reg = reg[["customer_id", "tenure", "monthly_charges", "total_charges"]]
    reg.total_charges = reg.total_charges.str.strip().replace('', np.nan).astype(float)
    reg = reg.dropna()
    return reg





# from env import host, user, password

# def get_db_url(db):
#     url = f'mysql+pymysql://{user}:{password}@{host}/{database_name}'
#     return url

# def get_data_from_mysql():
# 	query = '''
# 	SELECT *
# 	FROM customers
# 	JOIN internet_service_types USING (internet_service_type_id)
# 	WHERE internet_service_type_id = 3
# 	'''
# 	df = pd.read_sql(query,get_db_url('telco_churn'))
# 	return df

# def clean_data(df):
# 	df = df[["customer_id", "tenure", "monthly_charges", "total_charges"]]
# 	df.total_charges = df.total_charges.str.strip().replace('',np.nan).astype(float)
# 	df = df.dropna()
# 	return df

# def wrangle_telco():
# 	df = get_data_from_mysql()
# 	df = clean_data(df)
# 	return df
# 	# alternative
# 	# return clean_data(get_data_from_mysql())

# # import wrangle
# # wrangle.wrangle_telco()
