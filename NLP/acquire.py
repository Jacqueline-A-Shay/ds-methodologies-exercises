from requests import get
from bs4 import BeautifulSoup

def get_article(url):
    headers = {'User-Agent': 'JQ'}
    response = get(url, headers = headers)
    soup = BeautifulSoup(response.text)
    title = soup.select('h1')[0].get_text()
    content = soup.select('body')[0].get_text().strip()
    
    return {'title':title, 'content':content}

def get_blog_articles():
    urls = ['https://codeup.com/codeups-data-science-career-accelerator-is-here/',
        'https://codeup.com/data-science-myths/',
        'https://codeup.com/data-science-vs-data-analytics-whats-the-difference/',
        'https://codeup.com/10-tips-to-crush-it-at-the-sa-tech-job-fair/',
        'https://codeup.com/competitor-bootcamps-are-closing-is-the-model-in-danger/']
    output = []
    for url in urls:
        output.append(get_article(url))
    return output