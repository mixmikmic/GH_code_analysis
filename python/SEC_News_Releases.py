get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This one is used to load web pages
from urllib.request import urlopen
# This one is used to parse (extract information) from web pages
from bs4 import BeautifulSoup
# This one is to search/replace in text
import re
# This one is for textual analysis
import nltk 

# Say we want the 2016 and 2017 PR. We can see on the website that 
# there were 248 PR in 2017 and 283 PR in 2016.
# From that information, we build a list of PR to get.
pr_list = [(2017, x + 1) for x in range(248)] + [(2016, x + 1) for x in range(283)]
pr_list[-10:]

# In class, we'll limit ourselves to the first 60 in 2017.
pr_list = [(2017, x + 1) for x in range(60)]

pr_item = (2017, 1)

# First we load the page
url = 'https://www.sec.gov/news/press-release/' + str(pr_item[0]) + '-' + str(pr_item[1])
page = urlopen(url)

# Check the status. 200 means OK
page.status

# Parse the page content
soup = BeautifulSoup(page, 'html.parser')

# We can look at the page title. That might be an easier way to extract the news title
soup.title.contents

title = soup.title.contents[0][10:]
title

# Now let's find the date
soup.find('p', attrs={'class' : 'article-location-publishdate'})

loc_date_str = soup.find('p', attrs={'class' : 'article-location-publishdate'}).get_text()
loc_date_str

location = loc_date_str.split(',')[0].strip()
location

date_str = loc_date_str.split(',')[1].strip() + ', ' + loc_date_str.split(',')[2][:5].strip()
date_str

date = pd.to_datetime(date_str)
date

# Finally, let's get the content
body = soup.find('div', attrs={'class' : 'article-body'}).get_text()
print(body)

body

# We want to get plain text only, so we need to remove all the non-visible charaters 
# such as `\xa0` and `\n` and replace them by a space.

# The '\n' (return) is easy as it's a specific case
body = body.replace('\n', ' ')
body

# Handling `\xa0` and all the other characters like that that might appear need more machinery.
# For that, we use regular expressions: https://docs.python.org/3/library/re.html

body = re.sub(r'[^\x00-\x7F]+',' ', body)
body

# Finally, we package the result in a dictionary. Once we have all the press releases
# in a list of dictionaries, it's very easy to convert to a pandas DataFrame.

result = {'year': pr_item[0],
          'item': pr_item[1],
          'title': title,
          'location': location,
          'date': date,
          'body': body}
result

# Good, now we're ready to package as a function
def download_and_parse_pr(pr_item):
    # First we load the page
    url = 'https://www.sec.gov/news/press-release/' + str(pr_item[0]) + '-' + str(pr_item[1])
    
    try:
        page = urlopen(url)

        # Parse the page content
        soup = BeautifulSoup(page, 'html.parser')

        # Get title
        title = soup.title.contents[0][10:]

        # Get location and date
        loc_date_str = soup.find('p', attrs={'class' : 'article-location-publishdate'}).get_text()
        location = loc_date_str.split(',')[0].strip()
        date_str = loc_date_str.split(',')[1].strip() + ', ' + loc_date_str.split(',')[2][:5].strip()
        date = pd.to_datetime(date_str)

        # Get the content
        body = soup.find('div', attrs={'class' : 'article-body'}).get_text()
        body = body.replace('\n', ' ')
        body = re.sub(r'[^\x00-\x7F]+',' ', body)

        result = {'year': pr_item[0],
                  'item': pr_item[1],
                  'title': title,
                  'location': location,
                  'date': date,
                  'body': body}
        return result
    
    except Exception as e:
        print(str(pr_item[0]) + '-' + str(pr_item[1]) + ': ' + str(e))
        return None

# Now download all the pages
results = []

for pr_item in pr_list:
    results.append(download_and_parse_pr(pr_item))

# Package the result as dataframe
df_pr = pd.DataFrame([r for r in results if r is not None])

df_pr.head()

# We can rearrange the columns
df_pr = df_pr[['year', 'item', 'date', 'location', 'title', 'body']]
df_pr

text = df_pr[df_pr.item==1].iloc[0].body
text

# First, we want to extract all the words in the text.
# The regexp_tokenize line will convert everything to lower cap, keep only words (i.e. drop numbers and
# ponctuation) and split everything in tokens.
tokens = nltk.regexp_tokenize(text.lower(), '[a-z]+')
tokens[:10]

# NLTK has a lot of functions for analysis, for simple ones like this to very complex.
# Let's look at the 20 most frequent words
freq = nltk.FreqDist(tokens)
pd.Series(freq).sort_values(ascending=False).head(20)

# We typically want to remove the stop words (frequent
# words like "a" and "the")
# We'll use the default english corpus stopwords, but
# first we need to download them if we haven't already.
# If it's the first time running this code, uncomment
# the last line and run. Then download the "stopwords"
# corpora.
# Note that for our purposes this doesn't make a
# difference, so you can skip the filtering on stop
# words.

#nltk.download()

sr = nltk.corpus.stopwords.words('english')
sr[:10]

clean_tokens = []
for t in tokens:
    if t not in sr:
        clean_tokens.append(t)
clean_tokens[:10]

# We could actually do all that in one go:
tokens = []
for t in nltk.regexp_tokenize(text.lower(), '[a-z]+'):
    if t not in sr:
        tokens.append(t)
tokens[:10]

# Next we may want to stem the words (remove the ending)
# Note: not necessary for us, the LM dictionary is not stemmed.
stemmed_tokens = []
for t in clean_tokens:
    t = nltk.PorterStemmer().stem(t)
    stemmed_tokens.append(t)
stemmed_tokens[:10]

# Before we can do the sentiment analysis, we need to load the Loughran and MacDonald dictionary.
# See https://www3.nd.edu/~mcdonald/Word_Lists.html
lmdict = pd.read_excel('https://www3.nd.edu/~mcdonald/Word_Lists_files/LoughranMcDonald_MasterDictionary_2014.xlsx')

lmdict.head()

lmdict.tail()

# So there are roughly 85k word in there. What are some positive words?
lmdict[lmdict.Positive != 0].head()

# And negative ones?
lmdict[lmdict.Negative != 0].head()

# Ok, so the number in the columns is not a dummy (1), but the year it was added to the dictionary.
# We need to get a list of all the negative and positive words.
neg_words =  lmdict.loc[lmdict.Negative != 0, 'Word'].str.lower().unique()
pos_words =  lmdict.loc[lmdict.Positive != 0, 'Word'].str.lower().unique()

neg_words[:20]

pos_words[:20]

# Count the number of positive and negative words.
pos_count = 0
neg_count = 0
for t in tokens:
    if t in pos_words:
        pos_count += 1
    elif t in neg_words:
        neg_count += 1
print('Positive count: ' + str(pos_count))
print('Negative count: ' + str(neg_count))

# A crude measure of sentiment is the normalized difference between the
# number of positive and negative words
sentiment = (pos_count - neg_count)/(pos_count + neg_count)
sentiment

# Now that may cause problems if with detect no positive
# or negative words (division by zero). In that case,
# we can assume the text is neutral (sentiment = 0)
if (pos_count + neg_count) > 0:
    sentiment = (pos_count - neg_count)/(pos_count + neg_count)
else:
    sentiment = 0
sentiment

def compute_sentiment(text):
    # Tokenize and remove stop words
    tokens = []
    for t in nltk.regexp_tokenize(text.lower(), '[a-z]+'):
        if t not in sr:
            tokens.append(t)
    tokens[:10]
    
    # Count the number of positive and negative words.
    pos_count = 0
    neg_count = 0
    for t in tokens:
        if t in pos_words:
            pos_count += 1
        elif t in neg_words:
            neg_count += 1
            
    # Compute sentiment
    if (pos_count + neg_count) > 0:
        sentiment = (pos_count - neg_count)/(pos_count + neg_count)
    else:
        sentiment = 0
    return sentiment

# Test it
compute_sentiment(text)

# Now we can apply it to all our new releases
df_pr['sentiment'] = df_pr['body'].apply(compute_sentiment)

# Top 10 negative news
df_pr.sort_values('sentiment').iloc[:10]

# Top 10 positive news
df_pr.sort_values('sentiment', ascending=False).iloc[:10]



