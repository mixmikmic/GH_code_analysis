#cd where the data is 

get_ipython().magic('matplotlib inline')

import pandas as pd

file = pd.read_excel("20170330.xls")
file.head(1)

file.columns

len(file.columns)

file.shape

file.columns = ['Numbering', 'SchoolName','DepartmentName','Author','AuthorForeign','PaperName','PaperNameForeign',
               'Advisor','AdvisorForeign','OralComissioner','OralComissionerForeign','OralDate','DegreeCategory',
                'GraduationSchoolYear','GraduationYear','PublicationYear','Language','ChineseKeywords','ForeignKeywords',
                'Abstract','AbstractForeign','PapersDirectory','References','NumberPapers']

import nltk
from nltk.corpus import stopwords
from collections import Counter
import re

from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

#Get the stopwords
stops = set(stopwords.words("english"))

abstract = file[['AbstractForeign']]
abstract.head(1)

# Define a function to do the cleaning and preprocessing

def preprocessor( data ):
    # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", data) 
    

    # 2. Convert to lower case,
    lower = letters_only.lower()              
    
    # 3. remove unwanted characters, split into individual words
    regexText = (re.sub(r'(?<!\d)\.(?!\d)', '',' '.join(re.findall(r'[\w.]+',
             re.sub("'s|'re|'d|'ve|'t|\d+", " ", lower))))).split() 
    
    # 4. Remove 1 characters
    words = [word for word in regexText if len(word) > 1]
                     
    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

sample = abstract.AbstractForeign[0]
sample

sample = preprocessor(sample)
sample

def tokenizer_lemma_nostop(text):
    lem = WordNetLemmatizer()
    return [lem.lemmatize(w,"v") for w in re.split('\s+', text.strip())             if w not in stops and re.match('[a-zA-Z]+', w) and w != 'nan']

tokenizer_lemma_nostop(sample)

#Publication years 86 and 102 can be disregarded
file[(file.PublicationYear == 86) | (file.PublicationYear == 102)]

abstract = file[['DegreeCategory','DepartmentName','PublicationYear','AbstractForeign']]
abstract = abstract[abstract.PublicationYear != 86]
abstract = abstract[abstract.PublicationYear != 102]
abstract.head(1)

DepartmentName = list(set(abstract.DepartmentName))
DepartmentName

PublicationYear = [103,104,105]
PublicationYear

import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(ngram_range=(1, 1),
                        preprocessor=preprocessor,
                        tokenizer=tokenizer_lemma_nostop)

file1 = abstract[abstract.DepartmentName == '經營管理碩士在職專班'][abstract.PublicationYear ==  103]
file1['abstract'] = file1.AbstractForeign.map(lambda x : str(x))
file1.head(2)

data = file1.abstract

doc_bag = count.fit_transform(data).toarray()

bag_cnts = np.sum(doc_bag, axis=0)
bag_cnts.shape

print("[most frequent vocabularies]")
bag_cnts = np.sum(doc_bag, axis=0)

top = 20
# [::-1] reverses a list since sort is in ascending order
for tok, v in zip(count.inverse_transform(np.ones(bag_cnts.shape[0]))[0][bag_cnts.argsort()[::-1][:top]], np.sort(bag_cnts)[::-1][:top]):
    print(tok, v)

for year in PublicationYear:
    for department in DepartmentName:
        print(year, department)

import csv

count = CountVectorizer(ngram_range=(1, 1),
                        preprocessor=preprocessor,
                        tokenizer=tokenizer_lemma_nostop)

parsed = False

while not parsed:
        for year in PublicationYear:
            for department in DepartmentName:
                try:  
                    file = abstract[abstract.DepartmentName == department][abstract.PublicationYear == year]
                    file['abstract'] = file.AbstractForeign.map(lambda x : str(x))
                    data = file['abstract']
                    doc_bag = count.fit_transform(data).toarray()
                    bag_cnts = np.sum(doc_bag, axis=0)

                    top = 100

                    with open('wordcount.csv', 'a', newline='',encoding='utf-8-sig') as csvfile:

                        fieldnames = ['year','department','word', 'count']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        writer.writeheader()
                        for tok, v in zip(count.inverse_transform(np.ones(bag_cnts.shape[0]))[0][bag_cnts.argsort()[::-1][:top]], np.sort(bag_cnts)[::-1][:top]):
                            writer.writerow({'year':year,'department':department,'word': tok, 'count': v})
                            
    
                except ValueError:
                    print('Invalid value!')
                    continue
                    
        parsed = True  



