import pandas as pd
import numpy as np
import env

def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_data_from_mysql():
	query = """
    SELECT * 
    FROM customers
    JOIN internet_service_types USING (internet_service_type_id)
    JOIN contract_types USING (contract_type_id)
    JOIN payment_types USING (payment_type_id)
    """
	db = "telco_churn"
	df = pd.read_sql(query, get_db_url(db))
	return df

