import pandas as pd
import env

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

iris_querry = """
	SELECT * FROM measurements JOIN species USING(species_id)
	
	"""

def get_iris_data():
    return pd.read_sql(iris_query,get_connection('iris_db'))
