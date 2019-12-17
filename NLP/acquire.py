 # imports
from requests import get
from bs4 import BeautifulSoup
import os
import pandas as pd

# project specific imports
from acquire_codeup_blog import get_blog_posts
from acquire_news_articles import get_news_articles


def get_news_articles():
	# store into local csv file or make a new request if file not exist
    filename = 'inshorts_news_articles.csv'

    # check for presence of the file, or make a new request
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return make_new_request()

def get_articles_from_topic(url):
	# provide identifier when requesting webpage
    headers = {'user-agent': 'JQ'}
    # request
    response = get(url, headers=headers)
    # parse requested content w/ beautifulsoup
    soup = BeautifulSoup(response.content, 'html.parser')

    output = []
    # select the class news-card as a whole
    articles = soup.select(".news-card")
    # filter specific content & store into the variable
    for article in articles: 
        title = article.select("[itemprop='headline']")[0].get_text()
        content = article.select("[itemprop='articleBody']")[0].get_text()
        author = article.select(".author")[0].get_text()
        published_date = article.select(".time")[0]["content"]
        category = response.url.split("/")[-1]
        # store each part into a dictionary
        article_data = {
            'title': title,
            'content': content,
            'category': category,
            'author': author,
            'published_date': published_date,
        }
        # store each dictionary into 1 list
        output.append(article_data)


    return output


def make_new_request():
	# retrieve data from every target webpage
    urls = [
        "https://inshorts.com/en/read/business",
        "https://inshorts.com/en/read/sports",
        "https://inshorts.com/en/read/technology",
        "https://inshorts.com/en/read/entertainment"
    ]

    output = []
    
    for url in urls:
        # We use .extend in order to make a flat output list.
        output.extend(get_articles_from_topic(url))

    print("content")
    print(output)
    # store data into df
    df = pd.DataFrame(output)
    # store df into csv file
    df.to_csv('inshorts_news_articles.csv') 

    return df

 
