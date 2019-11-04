import pandas as pd
import env

# def get_connection(db, user=env.user, host=env.host, password=env.password):
#     return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_db_url(db, user= env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# Remove any properties that are likely to be something other than a single unit 
# properties (e.g. no duplexes, no land/lot, ...). There are multiple ways to 
# estimate that a property is a single unit, and there is not a single "right" 
# answer.

# def get_zillow_data():
#     query = '''
#     SELECT p2.*, p1.logerror FROM predictions_2016 p1
#         LEFT JOIN properties_2016 p2  USING(parcelid)
#         WHERE (bedroomcnt > 0 AND bathroomcnt > 0 AND calculatedfinishedsquarefeet > 500 
#             AND latitude IS NOT NULL AND longitude IS NOT NULL) 
#             AND (unitcnt = 1 OR unitcnt IS NULL);
#     '''
#     return pd.read_sql(query, get_connection('zillow'))

def get_zillow_data():
    query = '''
        select 
        svi.`COUNTY` county,
        p.`taxamount`/p.`taxvaluedollarcnt` tax_rate,
        p.`id`,
        p.`parcelid`,
        p.`airconditioningtypeid`,
        act.`airconditioningdesc`,
        p.`architecturalstyletypeid`,
        ast.`architecturalstyledesc`,
        p.`basementsqft`,
        p.`bathroomcnt`,
        p.`bedroomcnt`,
        p.`buildingclasstypeid`,
        bct.`buildingclassdesc`,
        p.`buildingqualitytypeid`,
        p.`calculatedbathnbr`,
        p.`calculatedfinishedsquarefeet`,
        p.`decktypeid`,
        p.`finishedfloor1squarefeet`,
        p.`finishedsquarefeet12`,
        p.`finishedsquarefeet13`,
        p.`finishedsquarefeet15`,
        p.`finishedsquarefeet50`,
        p.`finishedsquarefeet6`,
        p.`fips`,
        svi.`ST_ABBR` state,
        p.`fireplacecnt`,
        p.`fullbathcnt`,
        p.`garagecarcnt`,
        p.`garagetotalsqft`,
        p.`hashottuborspa`,
        p.`heatingorsystemtypeid`,
        hst.`heatingorsystemdesc`,
        p.`latitude`,
        p.`longitude`,
        p.`lotsizesquarefeet`,
        p.`poolcnt`,
        p.`poolsizesum`,
        p.`pooltypeid10`,
        p.`pooltypeid2`,
        p.`pooltypeid7`,
        p.`propertycountylandusecode`,
        p.`propertylandusetypeid`,
        plut.`propertylandusedesc`,
        p.`propertyzoningdesc`,
        p.`rawcensustractandblock`,
        p.`regionidcity`,
        p.`regionidcounty`,
        p.`regionidneighborhood`,
        p.`regionidzip`,
        p.`roomcnt`,
        p.`storytypeid`,
        st.`storydesc`,
        p.`taxvaluedollarcnt`,
        p.`threequarterbathnbr`,
        p.`unitcnt`,
        p.`yardbuildingsqft17`,
        p.`yardbuildingsqft26`,
        p.`yearbuilt`,
        p.`numberofstories`,
        p.`fireplaceflag`,
        p.`structuretaxvaluedollarcnt`,
        p.`assessmentyear`,
        p.`landtaxvaluedollarcnt`,
        p.`taxamount`,
        p.`taxdelinquencyflag`,
        p.`taxdelinquencyyear`, 
        p.`typeconstructiontypeid`,
        tct.`typeconstructiondesc`,
        p.`censustractandblock`,
        pred.`transactiondate`,
        pred.`logerror`,
        m.`transactions`
    from 
        `properties_2017` p
    inner join `predictions_2017`  pred
        on p.`parcelid` = pred.`parcelid` 
    inner join 
        (select 
            `parcelid`, 
            max(`transactiondate`) `lasttransactiondate`, 
            max(`id`) `maxid`, 
            count(*) `transactions`
        from 
            predictions_2017
        group by 
            `parcelid`
        ) m
        on 
        pred.parcelid = m.parcelid
        and pred.transactiondate = m.lasttransactiondate
    left join `propertylandusetype` plut
        on p.`propertylandusetypeid` = plut.`propertylandusetypeid`
            
    left join svi_db.svi2016_us_county svi
        on p.`fips` = svi.`FIPS`
    left join `airconditioningtype` act
        using(`airconditioningtypeid`)
    left join heatingorsystemtype hst
        using(`heatingorsystemtypeid`)
    left join `architecturalstyletype` ast
        using(`architecturalstyletypeid`)
    left join `buildingclasstype` bct
        using(`buildingclasstypeid`)
    left join `storytype` st
        using(`storytypeid`)
    left join `typeconstructiontype` tct
        using(`typeconstructiontypeid`)
    where 
        p.`latitude` is not null
        and p.`longitude` is not null;
        '''

    df = pd.read_sql(query, get_db_url('zillow'))
    return df

def get_iris_data():
    query = '''
    SELECT petal_length, petal_width, sepal_length, sepal_width, species_id, species_name
    FROM measurements m
    JOIN species s USING(species_id)
    '''
    return pd.read_sql(query, get_connection('iris_db'))

def get_mallcustomer_data():
    df = pd.read_sql('SELECT * FROM customers;', get_connection('mall_customers'))
    return df.set_index('customer_id')