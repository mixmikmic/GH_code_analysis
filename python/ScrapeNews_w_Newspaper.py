# obtain a article and parse it
from newspaper import Article
url = 'http://www.cnn.com/2017/07/18/politics/obamacare-fail-trump/index.html'
article = Article(url)
article.download()
#article.html
article.parse()

# print article infomation
print(article.title)
print(article.authors)
print(article.publish_date)

# print out article body
print(article.text)

print(article.top_image)

article.nlp()

# find keywords in the article
print(article.keywords)

# print summary in the article
print(article.summary)

import newspaper

cnn_articles = newspaper.build('http://cnn.com',memoize_articles=False)

for item in cnn_articles.articles:  
    print(item.url)

for category in cnn_articles.category_urls(): 
    print(category)

# check one of the article
cnn_article = cnn_articles.articles[5]
cnn_article.download()
cnn_article.parse()
print(cnn_article.title)

cnn_politics_articles = newspaper.build('http://www.cnn.com/politics',memoize_articles=False)

for item in cnn_politics_articles.articles:  
    print(item.url)

# print out the number of articles
print(cnn_politics_articles.size())

# print out information of one article
politics_article = cnn_politics_articles.articles[100]
politics_article.download()
politics_article.parse()
print(politics_article.title)

import pandas as pd

col_names = ["source", "title", "author", "text"]
article_df = pd.DataFrame(columns = col_names)

for i in range(0,20):
    a = cnn_politics_articles.articles[i];
    a.download()
    a.parse()   
    entry = pd.DataFrame([["cnn", a.title,a.authors,a.text]],index = [i],columns=col_names)
    article_df = article_df.append(entry) 
    
article_df  

article_df.to_csv("data_news.csv")

source_list = []
title_list = []
author_list = []
text_list = []

for i in range(0,20):
    a = cnn_politics_articles.articles[i];
    a.download()
    a.parse()     
    source_list.append("http://www.cnn.com/politics");
    title_list.append(a.title);
    author_list.append(a.authors);
    text_list.append(a.text);
        
article_df = pd.DataFrame(source_list, columns = ["source"])   
article_df["title"] = title_list
article_df["author"] = author_list
article_df["text"] = text_list

article_df.head()

