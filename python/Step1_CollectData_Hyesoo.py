hyesoo_fake_news = ['http://ABCnews.com.co','http://Americannews.com','http://Americanoverlook.com', 
                    'http://Bighairynews.com','http://bizstandardnews.com','http://Bloomberg.ma',
                    'http://70news.wordpress.com','http://beforeitsnews.com','http://Cap-news.com',
                    'http://ddsnewstrend.com', 'http://thebostontribune.com/','http://americanfreepress.net/',
                    'http://www.bipartisanreport.com/','http://aurora-news.us/', 'http://Clashdaily.com',
                    'http://Conservativedailypost.com', 'http://Conservativeinfidel.com','http://Dailyheadlines.com',
                    'http://DeadlyClear.wordpress.com', 'http://Donaldtrumpnews.co', 'http://Freedomdaily.com']


hyesoo_true_news = ['https://www.nytimes.com/','http://www.bbc.com/news',
             'http://www.npr.org/sections/news/', 'http://www.reuters.com/',
             'https://www.apnews.com/', 'http://www.cnn.com', 'http://www.foxnews.com/', 
             'http://www.politico.com/'] 

print("We have {} fake news and {} true news organizations".format(len(hyesoo_fake_news), len(hyesoo_true_news)))

import pandas as pd
import newspaper 
from itertools import islice
import os

def generate_raw_true_data(news_list, data_name):

    col_names = ["url","source", "title", "author", "text"]
    article_df = pd.DataFrame(columns = col_names)
    final_news_list = []
    final_article_number = {}
    total_count = 0
    for news in news_list:
        try:
            news_articles = newspaper.build(news, memoize_articles=False)
            final_news_list += [news_articles]
        except:
            pass
    print ([a.brand for a in final_news_list], len(final_news_list))


    for news_articles in final_news_list:
        count = 0
        num = len([x for x in news_articles.articles])
        if num >= 250:
            news_articles_articles = news_articles.articles[:250]
        else:
            news_articles_articles = news_articles.articles
        for article in news_articles_articles:
            try:
                article.download()
                article.parse()
                entry = pd.DataFrame([[article.url, news_articles.brand, article.title, article.authors, article.text]], columns=col_names)                    
                article_df = article_df.append(entry)
                count += 1
                total_count += 1
                print(article.url)
            except:
                pass
        print("The total number of " + str(news_articles.brand) + " articles is ", count) 
        final_article_number[news_articles.brand] = count

    print(total_count)
    path = os.path.join('data', data_name+'.csv')
    article_df.to_csv(path)
    return final_article_number

# true
generate_raw_true_data(hyesoo_true_news, 'hyesoo_true_news_rawdata')

def generate_raw_fake_data(news_list, data_name):
    col_names = ["url", "source", "title", "author", "text"]
    article_df = pd.DataFrame(columns = col_names)
    final_news_list = []
    final_article_number = {}
    total_count = 0
    for news in news_list:
        try:
            news_articles = newspaper.build(news, memoize_articles=False)
            final_news_list += [news_articles]
        except:
            pass
    print (final_news_list, len(final_news_list))
    for news_articles in final_news_list:
        if total_count < 1200:
            count = 0
            for article in news_articles.articles:
                try:
                    article.download()
                    article.parse()
                    entry = pd.DataFrame([[article.url, news_articles.brand, article.title, article.authors, article.text]], columns=col_names)
                    article_df = article_df.append(entry)
                    count += 1
                    total_count += 1
                    print(article.url)
                except:
                    pass
            print("The total number of " + str(news_articles.brand) + " articles is ", count) 
            final_article_number[news_articles.brand] = count
        else:
            pass
    print(total_count)
    article_df.to_csv(data_name+".csv")
    return final_article_number

# fake
generate_raw_fake_data(hyesoo_fake_news, 'hyesoo_fake_news_rawdata')



