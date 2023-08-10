### EDA packages and Web
import pandas as pd
import lxml
import requests

url_list = ["https://www.reuters.com/news/health",
            "https://www.reuters.com/politics",
            "https://www.reuters.com/finance",
            "https://www.reuters.com/news/sports",
            "https://www.reuters.com/news/technology"]
            
feeds_list = [
            "http://feeds.reuters.com/reuters/businessNews",
            "http://feeds.reuters.com/reuters/technologyNews",
            "http://feeds.reuters.com/reuters/sportsNews",
            "http://feeds.reuters.com/reuters/healthNews",
            "http://feeds.reuters.com/reuters/politicsNews",]

# Using LXML 
from lxml import etree

# Scraping and Parsing Data From Feeds_list
datafeeds = []
for feed in feeds_list:
    response = requests.get(feed)
    xml_page = response.text
    parser = etree.XMLParser(recover=True, encoding='utf-8')
    datafeeds.append(etree.fromstring(xml_page.encode("utf-8"), parser=parser))

# Function for Building Node
def print_tag(node):
    print("<%s %s>%s" % (node.tag, " ".join(["%s=%s" % (k,v)for k,v in node.attrib.iteritems()]), node.text))
    for item in node[:25]:
        print("  <%s %s>%s</%s>" % (item.tag, " ".join(["%s=%s" % (k,v)for k,v in item.attrib.iteritems()]), item.text, item.tag))
    print("</%s>" % node.tag)


# What we want to select
general_node = datafeeds[0]
print_tag(general_node)

# Selecting Node
general_node = general_node[0]
print_tag(general_node)

# Specific Selection of Item
general_node = general_node.xpath("item")[0]
print_tag(general_node)

# Grouping them  into List and Array
title_list = []
description_list = []
category_list = []

for xml_doc in datafeeds:
    articles = xml_doc.xpath("//item")
    for article in articles: #0,1,4 instead of 0,2,3
        title_list.append(article[0].text)
        description_list.append(article[1].text)
        category_list.append(article[4].text)
        


# Putting Data Into DataFrame
news_df = pd.DataFrame(title_list, columns=["Title"])
news_df["Description"] = description_list
news_df["Category"] = category_list
print(len(news_df))
news_df


news_df["Description"].head()

print(news_df["Description"][0])

get_ipython().run_cell_magic('HTML', '', 'NEW YORK (Reuters) - Financial stocks led a drop on Wall Street on Friday, as results from big banks failed to enthuse and geopolitical tensions in Syria and Russia further unnerved investors.<div class="feedflare">\n<a href="http://feeds.reuters.com/~ff/reuters/businessNews?a=G0SWmsbH8M8:33ar9b0m6EQ:yIl2AUoC8zA"><img src="http://feeds.feedburner.com/~ff/reuters/businessNews?d=yIl2AUoC8zA" border="0"></img></a> <a href="http://feeds.reuters.com/~ff/reuters/businessNews?a=G0SWmsbH8M8:33ar9b0m6EQ:F7zBnMyn0Lo"><img src="http://feeds.feedburner.com/~ff/reuters/businessNews?i=G0SWmsbH8M8:33ar9b0m6EQ:F7zBnMyn0Lo" border="0"></img></a> <a href="http://feeds.reuters.com/~ff/reuters/businessNews?a=G0SWmsbH8M8:33ar9b0m6EQ:V_sGLiPBpWU"><img src="http://feeds.feedburner.com/~ff/reuters/businessNews?i=G0SWmsbH8M8:33ar9b0m6EQ:V_sGLiPBpWU" border="0"></img></a>\n</div><img src="http://feeds.feedburner.com/~r/reuters/businessNews/~4/G0SWmsbH8M8" height="1" width="1" alt=""/>')

# Create A Short Description
news_df["Short_description"] = [item[item.find(" - ")+3:item.find("<")] for item in news_df["Description"]]
news_df

news_df

# Save to A CSV File
news_df.to_csv("ReutersNewsDataFinal2.csv")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB

corpus = news_df["Short_description"]
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus).toarray()

# Shape of Our Data
print(X.shape)

# Features
X

# Names of Vectorized Features
vectorizer.get_feature_names()[:25]

# Building a Map of Categories =Making Categories Numerical since ML understands numbers better
categories = news_df["Category"].unique()
category_dict = {value:index for index, value in enumerate(categories)}
results = news_df["Category"].map(category_dict)
category_dict


print("corpus size: %s" % len(vectorizer.get_feature_names()))

# Labels
results

# Split Dataset into Test and Training Data
x_train,x_test, y_train,y_test = train_test_split(X, results, test_size=0.2, random_state=1, )

# Using NaiveBaiyes Multinomial Classifier
clf = MultinomialNB()
clf.fit(x_train, y_train)

print("Accuracy of our model score: ",clf.score(x_test, y_test))

clf.predict(x_test)

category_dict

### Sample Prediction of Category of News

text = ["Russian Hackers hijack US Election "]

# Vectorize and Transform text
vec_text = vectorizer.transform(text).toarray()

# Predict
clf.predict(vec_text)

#category_dict.keys()[category_dict.values().index(clf.predict(vec_text)[0])]

# A function to do it
def newscategorifier(a):
    test_name1 = [a]
    transform_vect =vectorizer.transform(text).toarray()
    if clf.predict(transform_vect) == 0:
        print("Business News")
    elif clf.predict(transform_vect) == 1:
        print("Technology News")
    elif clf.predict(transform_vect) == 2:
        print("Sport News")
    elif clf.predict(transform_vect) == 3:
        print("Health News")
    else:
        print("Politcs News")

newscategorifier("Python and Julia for Computer Scientist")

## Save Our Model to be used
from sklearn.externals import joblib

NaiveBayModel = open("newsclassifierNBmodel.pkl","wb")

joblib.dump(clf,NaiveBayModel)

## Thanks For Watching
# J-Secur1ty by Jesse
# Jesus Saves @ JCharisTech



