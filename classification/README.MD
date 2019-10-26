# ds-methodologies-exercises

**1. acquire data**

- Codeup: 

> setup: acquire.py, env.py
>
> import pandas as pd
>
> df = pd.DataFrame(acquire.get_data_from_mysql()) 
>
> you will be inputing specific SQL query and database (titanic_db or iris_db)

- outside of Codeup: 

> setup: pydataset

> from pydataset import data # obtain datasets
>
> titanic = data('titanic') # obtain 'titanic' data
>
> iris = data('iris')
>
> data('titanic', show_doc=True) # view documentation

