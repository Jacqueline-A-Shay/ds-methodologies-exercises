def acquire():
    import requests
    import pandas as pd    
    for page in range(1,5):
        base_url = 'https://python.zach.lol/api/v1/sales'
        response = requests.get(base_url + '?page=' + str(page))
        sales = pd.DataFrame()
        sales = pd.concat([sales, pd.DataFrame(response.json()['payload']['sales'])])
    return sales 