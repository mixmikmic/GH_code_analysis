# url of the news organizations, in a form of dictionary, eg. CNN : https://www.CNN.com
# May use a cvs file and open/read the file
# news_org_list = ['https://www.CNN.com', 'http://www.bbc.com/news']
news_org_list = ['http://www.cnn.com/2017/07/21/politics/wapo-sessions-discussed-campaign-with-kislyak/index.html']

# Import the packages
import newspaper as np
import pandas as pd

# obtain the articles from each news organization and make a dictionary. eg) 'CNN' : 'https://www.CNN.com'

all_news = {}
news_name_list = []
for i in range(len(news_org_list)):
    news = np.build(news_org_list[i],memoize_articles=False)
    news_name_list += [news.brand]
    all_news[news.brand] = news

# Detail information about each article
# Given that each list will consist of information with the same article order, make the lists of each information

# lists of each information that are useful & we can get from Newspaper package
urls = []
text_body = []
keywords = []
date = []
unique_words_freq = []
source = []
authors = []


num_news_org = len(news_org_list)

for n in range(num_news_org):
    news_org_name = news_name_list[n]
    news = all_news[news_org_name]
    for article in news.articles:
        urls += [article.url]
        source += [news.brand]
        article.download()
        article.parse()
        authors += [article.authors]
        date += [article.publish_date]
        text_body += [article.text]

# #         print(article.url)

text_body[0]

# Basic function for parsing the text
# for automation, inputs should be a series of articles





# combining alphabets into a word 
# Return a word in a str format
def combine_words(word):
    if len(word) == 1:
        return word[0]
    else:
        return word[0] + combine_words(word[1:])

    
    
    
# changing sentences into a list of words
# Removing common / less meaningful words
# Return 2 values: the words list, and a dictionary of each_unique_word : frequency
def parsing (text):
    parsed_text = list(text)
    # Needs to be updated
    avoid_list = [' ', ',', '.', '?', '!', '~', '*','#','\n','-','(',')','"',"'",':']
    if parsed_text[-1] in avoid_list:
        parsed_text = parsed_text[:-1]
    n = len(parsed_text)
    word = []
    i=0
    final_words = []

    while i < n:
        if parsed_text[i] not in avoid_list:
            word += [parsed_text[i]]
            if i == n-1:
                final_words += [combine_words(word)]
        else:
            if parsed_text[i-1] in avoid_list:
                pass
            else:
                final_words += [combine_words(word)]
                word = []
        i += 1
    # Probably can get a better list of commonly used words
    common_words = ['an','just','them','as','says','their','by','that','the','have','they','we','with','he','she','him','her','in','a','and','on','are','of','from','what','to','for','it','out','is','does','were','was','am','I','you','he','our']
    final_words = [x.lower() for x in final_words]
    final_words = [x for x in final_words if x not in common_words and x !='s']
    return (final_words, dict((x,final_words.count(x)) for x in set(final_words)))

collected_words, unique_words = parsing(text_body[0])
unique_words

# This function returns the number of unique words per the words in the article, 
# and a dictionary of most frequent words and their number

def unique_words_info(dic, num):
    freq_unique_words = len(dic)/sum(dic.values())
    freq_words = {k:v for (k,v) in unique_words.items() if v > num}
    return (freq_unique_words, sorted(freq_words.items(), key=lambda x: x[1]))


unique_words_info(unique_words, 8)

# Can check if it contains any fish words in our list
# Returns a binary list

fishy_words = ['wow','niger','tip','fake','whatever']
def binary(dic, fishy_words):
    fishy_words = fishy_words
    binary_result = [0] * len(fishy_words)
    for i in range(len(fishy_words)):
        if fishy_words[i] in dic.keys():
            binary_result[i] = 1
    return binary_result

binary(unique_words, fishy_words)



