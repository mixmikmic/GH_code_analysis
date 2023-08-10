import numpy as np
import pandas as pd
import re
import string
#import pyspark

search_data = pd.read_csv("./data/sample_openData_searchTerms_clean.csv")

search_data.shape

search_data.tail()

#search_data[search_data['Total Unique Searches']<5]

search_terms = list(set(search_data['Search Term']))

len(search_terms)

#print search_terms

search_data.ix[search_data["Search Term"] == "194415"]

search_terms_data = search_data[["Search Term"]]
search_terms_data = search_terms_data.rename(columns={"Search Term": "search_term"})

search_terms_data.head()

search_terms_data['processed_data'] = search_terms_data.search_term                                        .apply(lambda text: text.decode('ascii' ,"ignore" ).decode('utf-8','ignore'))                                        .apply(lambda text: text.lower())

search_terms_list =  list(set(search_terms_data.processed_data))
#print search_terms_list

dates_r = re.compile('[0-9]/[0-9]/[0-9]')
numbers_r = re.compile('^[0-9][0-9]*[0-9]$')
html_r = re.compile('^<.*>$')

filter(dates_r.match, search_terms_list)

print filter(html_r.match, search_terms_list)

# removing punctuation

def removePunctuation(text):

    for c in string.punctuation:
        text = text.replace(c,"").strip().lower()
    return text

# iterative process

def text_processing(search):
    
    return [removePunctuation(i) for i in search]
        
        

search = text_processing(search_terms_list)
#search

regex = "\d{1,4}.?\d{0,4}\s[a-zA-Z|\d+]{2,30}\s[a-zA-Z]|\s[a-zA-Z]*"

f = [re.findall(regex, i) for i in search
     if re.findall(regex, i)!= [] 
     if re.findall(regex, i)[0][:3] != '311'
     ]

# http://regexlib.com/REDetails.aspx?regexp_id=430

year = [str(j) for j in range(2000,2017)]

addresses = [i for i in f if i[0][:4] not in year ]
#addresses

# applying existing code to the full data

query = pd.read_csv("./data/all_queries.csv")

query.head()

len(query)

query.shape

# seeing the data

query_list =  list(query['ga.searchKeyword'].values)
query_list = [str(word).decode('ascii' ,"ignore" ).decode('utf-8','ignore') for word in query_list]

set_query = len(set(query_list))
set_query

#adresses
regex = "\d{1,4}.?\d{0,4}\s[a-zA-Z|\d+|\W+]{2,30}\s[a-zA-Z]{2,15}"

full = [re.findall(regex, i) for i in query_list
     if re.findall(regex, i)!= [] 
     if re.findall(regex, i)[0][:3] != '311'
     ]



def search_term_type(text):
    
    """
    input: keyword search term
    output: labels the keyword as good, address, year, link.
    
    """
    
    regex = "\d{1,4}.?\d{0,4}\s[a-zA-Z|\d+|\W+]{2,30}\s[a-zA-Z]{2,15}"
    links = "((https?|http):((//)|(\\\\))+[\w\d:#@%/;$()~_?\+-=\\\.&]*)|www.*"
    #link = '[\w\d:#@%/;$()~_?\+-=\\\.&]*'
    years = [str(i) for i in range(2000,2017)]
    if len(re.findall(regex, text))> 0 and re.findall(regex, text)[0][:4] not in years and re.findall(regex, text)[0][:3] != '311':
        return "Address"
    if len(re.findall(links, text))> 0:
        return "Link"
    if len(re.findall(regex, text))> 0 and re.findall(regex, text)[0][:4] in years:
        return "Year"
    
    return "Valid Search Term"

'311 census '

years = [str(i) for i in range(2000,2017)]

string = 'data for crime'
search_term_type(string)

query['Search Type'] = ['blank'] * len(query)

keywords = list(query['ga.searchKeyword'].values)

types = [search_term_type(str(word)) for word in keywords]

query['Search Type'] = types

query.to_csv('all_queries_w_search_type')

Address = query[query['Search Type'] == 'Valid Search Term']
Address 

Address['ga.searchAfterDestinationPage'].tolist()

text = '2010 us census data'
regex = "\d{1,4}.?\d{0,4}\s[a-zA-Z|\d+|\W+]{2,30}\s[a-zA-Z]{2,15}"
r = re.findall(regex, text)[0][:4]
r in range(2000,2017)



# flatten full list
full = [j for i in full for j in i]
full

year_full = [str(j) for j in range(2000,2017)]
# taking out the years

addresses_full = [i for i in full if i[:4] not in year ]
addresses_full = list(set(addresses_full))

year_listing = [i.lower() for i in full if i[:4] in year ]
year_listing = list(set(year_listing))
year_listing

#clean_list = sc.parallelize(query_list).filter(lambda word: word.lower() not in addresses_full).collect()

clean_list = [i for i in query_list if i.lower() not in addresses_full]
#clean_list

links = "((https?|http):((//)|(\\\\))+[\w\d:#@%/;$()~_?\+-=\\\.&]*)"

https = [re.findall(links, i) for i in clean_list
     if re.findall(links, i)!= [] 
     if re.findall(links, i)!= "//" 
     ]
# http://stackoverflow.com/questions/6718633/python-regular-expression-again-match-url

# finding all the unique links that people have put into the search query

# https = (sc.parallelize(https).flatMap(lambda word: word)
#         .flatMap(lambda word: word)
#         .filter(lambda word: word != '//')
#         .filter(lambda word: word != 'https')
#         .filter(lambda word: word != '''''')
#         .filter(lambda word: word != 'http')
#         .collect())

https = [i[0] for i in https]

https

# clean_list = (sc.parallelize(query_list).map(lambda word: word.lower()).filter(lambda word: word not in https)
#               .filter(lambda word: word not in addresses_full)
#               .collect())

clean_list = [word for word in query_list if word not in https if word not in addresses_full]

from collections import Counter
word_count = Counter(clean_list)
word_count_sorted = sorted(word_count.items(),key = lambda x: x[1], reverse=True)

word_count_sorted = [i[0] for i in word_count_sorted if i[1]>2]
word_count_sorted

string_word = removePunctuation(re.sub("\u" , '', str(word_count_sorted)))
string_word1 = removePunctuation(re.sub("\u" , '', str(query_list)))

from polyglot.text import Text

NER = Text(string_word1)

NER = NER.entities

for entity in NER:
    if entity.tag == "I-PER":
        print entity
    

#not_caught = [ '3180 18th street' , '100 church', '17th street' , '800 university avenue, palo alto, california', '17 san andreas way, san francisco' , '2631 23rd']

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF

vec=CountVectorizer(stop_words='english',analyzer='word')
X_train_counts = vec.fit_transform(word_count_sorted)
vocab = vec.get_feature_names()
nmf = NMF(n_components = 10, random_state = 1)
nmf.fit(X_train_counts)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

print(print_top_words(nmf, clean_list, 50))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(word_count_sorted)
indices = np.argsort(vectorizer.idf_)
features = vectorizer.get_feature_names()
top_n = 100
top_features = [features[i] for i in indices[:top_n]] 

print top_features

import pickle

pickle.dump(word_count_sorted , open("word_count_sorted.p",'wb'))

