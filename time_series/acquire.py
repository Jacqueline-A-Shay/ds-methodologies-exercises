def acquire(base_url):
    p = 2
    import requests
    import pandas as pd
    response = requests.get(base_url)
    sale_data = response.json()
    sales = pd.DataFrame(sale_data['payload']['sales'])
    while sale_data['payload']['page'] < 183:
        response = requests.get(base_url + 'page =' + str(p))
        sales = pd.concat([sales, pd.DataFrame(sale_data['payload']['sales'])])
        p += 1
        if p > sale_data['payload']['max_page']:
            sales = sales.reset_index()
            return sales         