import pandas as pd
import numpy as np
import collections 
from collections import Counter
from nltk.corpus import stopwords

document = pd.read_csv('datadump.csv', delimiter="|", error_bad_lines=False)

df = document[['cc','title','tags']]

df.head()

us_dataframe = df.loc[df['cc'] == 'en-us'] 
au_dataframe = df.loc[df['cc'] == 'en-au'] 
in_dataframe = df.loc[df['cc'] == 'en-in'] 
ca_dataframe = df.loc[df['cc'] == 'en-ca'] 
uk_dataframe = df.loc[df['cc'] == 'en-uk'] 

#us_dataframe = us_dataframe.drop_duplicates('id')
#au_dataframe = au_dataframe.drop_duplicates('id')
#in_dataframe = in_dataframe.drop_duplicates('id')
#ca_dataframe = ca_dataframe.drop_duplicates('id')
#uk_dataframe = uk_dataframe.drop_duplicates('id')

us_dataframe['tag_string'] = us_dataframe['tags'].astype(str)
au_dataframe['tag_string'] = au_dataframe['tags'].astype(str)
in_dataframe['tag_string'] = in_dataframe['tags'].astype(str)
ca_dataframe['tag_string'] = ca_dataframe['tags'].astype(str)
uk_dataframe['tag_string'] = uk_dataframe['tags'].astype(str)

us_dataframe.head(3)

uk_title = []
uk_tags = []

us_title = []
us_tags = []

in_title = []
in_tags = []

au_title = []
au_tags = []

ca_title = []
ca_tags = []

for index, row in us_dataframe.iterrows():
    stringer = row[1]
    parsed = stringer.split()
    for word in parsed:
        us_title.append(word)
for index, row in uk_dataframe.iterrows():
    stringer = row[1]
    parsed = stringer.split()
    for word in parsed:
        uk_title.append(word)
for index, row in ca_dataframe.iterrows():
    stringer = row[1]
    parsed = stringer.split()
    for word in parsed:
        ca_title.append(word)
for index, row in in_dataframe.iterrows():
    stringer = row[1]
    parsed = stringer.split()
    for word in parsed:
        in_title.append(word)
for index, row in au_dataframe.iterrows():
    stringer = row[1]
    parsed = stringer.split()
    for word in parsed:
        au_title.append(word)

for index, row in us_dataframe.iterrows():
    stringer = row[3]
    parsed = stringer.split()
    for word in parsed:
        us_tags.append(word)
for index, row in uk_dataframe.iterrows():
    stringer = row[3]
    parsed = stringer.split()
    for word in parsed:
        uk_tags.append(word)
for index, row in au_dataframe.iterrows():
    stringer = row[3]
    parsed = stringer.split()
    for word in parsed:
        au_tags.append(word)
for index, row in in_dataframe.iterrows():
    stringer = row[3]
    parsed = stringer.split()
    for word in parsed:
        in_tags.append(word)
for index, row in ca_dataframe.iterrows():
    stringer = row[3]
    parsed = stringer.split()
    for word in parsed:
        ca_tags.append(word)

def counter_filter(words):  
    
    for_counting = []
   
    stop_words = ['globaleg','nan','c','b','a','d','The','the', 'To','to','Of','of','A','Are','This','That','And','In','Is', 'For','On','With','Which','What']
    
    
    # set to lowercase
    
        
    # filter words
    for word in words:
        if word not in stop_words:
            for_counting.append(word)
    # Start counting
    word_count = Counter(for_counting)

        # The Top-N words
    top_list = word_count.most_common(30)
    print top_list
    top_justwords = []
    for i in top_list:
        top_justwords.append(i[0])
    
    print for_counting
       
    # create file    
   
    with open('forvizualization.txt', 'w') as f:
        for item in for_counting:
                f.write("%s\n" % item)
    


counter_filter(us_title)


