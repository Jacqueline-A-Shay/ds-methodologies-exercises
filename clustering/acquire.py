import pandas as pd
import numpy as np
import env

def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_data_from_mysql():

	query = """
	
	SELECT 

	svi.county county,
	svi.st_abbr state,
	
	pr.parcelid as pp, 
	pr.logerror, 
	pr.transactiondate, 
	p.*,
	a.airconditioningdesc,
	ar.architecturalstyledesc,
	b.buildingclassdesc,
	h.heatingorsystemdesc,
	prop.propertylandusedesc,
	s.storydesc,
	t.typeconstructiondesc


	FROM predictions_2017 pr
	
	JOIN 
	(SELECT parcelid as drop_parcelid, MAX(transactiondate) AS tdate 
	FROM predictions_2017
	GROUP BY drop_parcelid)   as md

	ON (md.drop_parcelid = pr.parcelid AND md.tdate = pr.transactiondate)
	
	JOIN properties_2017 p ON pr.parcelid = p.parcelid
	
	LEFT JOIN airconditioningtype a USING (airconditioningtypeid)
	LEFT JOIN architecturalstyletype ar USING (architecturalstyletypeid)
	LEFT JOIN buildingclasstype b USING (buildingclasstypeid)
	LEFT JOIN heatingorsystemtype h USING (heatingorsystemtypeid)
	LEFT JOIN propertylandusetype prop USING (propertylandusetypeid)
	LEFT JOIN storytype s USING (storytypeid)
	LEFT JOIN typeconstructiontype t USING (typeconstructiontypeid)
	LEFT JOIN svi_db.svi2016_us_county svi ON svi.fips = p.fips
	WHERE (p.latitude IS NOT NULL AND p.longitude IS NOT NULL);	

    """
	db = "zillow"
	df = pd.read_sql(query, get_db_url(db))

	return df

def basic_clean(df):
	df = df.drop(\
		columns = ["pp", "airconditioningtypeid","architecturalstyletypeid",\
		"buildingclasstypeid", "heatingorsystemtypeid","typeconstructiontypeid"], axis = 1)\
	.set_index("parcelid")
	return df

def sum_data(df):
	print("numeric data description: {}".format(df.describe()))
	print("all data description: {}".format(df.describe(include='all')))
	print(df.info())
	print(df.shape)
	print(df.isnull().sum())

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    pct_missing = num_missing/rows
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing,\
     'pct_rows_missing': pct_missing})\
    .sort_values(by = 'pct_rows_missing', ascending = False)
    return cols_missing

def nulls_by_row(df):
    num_cols_missing = df.isnull().sum(axis=1)
    pct_cols_missing = df.isnull().sum(axis=1)/df.shape[1]*100
    rows_missing = pd.DataFrame({'num_cols_missing': num_cols_missing,\
                                 'pct_cols_missing': pct_cols_missing})\
                                .reset_index()\
                                .groupby(['num_cols_missing','pct_cols_missing'])\
                                .count()\
                                .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing





