# We use the following magic commands to time the cells in the notebook
get_ipython().magic('install_ext https://raw.github.com/cpcloud/ipython-autotime/master/autotime.py')
get_ipython().magic('load_ext autotime')

from os import listdir, chdir
import re

# Defining regular expressions 

re0 = re.compile('>')
re1 = re.compile('(Message-ID(.*?\n)*X-FileName.*?\n)|'
                 '(To:(.*?\n)*?Subject.*?\n)|'
                 '(< (Message-ID(.*?\n)*.*?X-FileName.*?\n))')
re2 = re.compile('(.+)@(.+)') # Remove emails
re3 = re.compile('\s(-----)(.*?)(-----)\s', re.DOTALL)
re4 = re.compile('''\s(\*\*\*\*\*)(.*?)(\*\*\*\*\*)\s''', re.DOTALL)
re5 = re.compile('\s(_____)(.*?)(_____)\s', re.DOTALL)
re6 = re.compile('\n( )*-.*')
re7 = re.compile('\n( )*\d.*')
re8 = re.compile('(\n( )*[\w]+($|( )*\n))|(\n( )*(\w)+(\s)+(\w)+(( )*\n)|$)|(\n( )*(\w)+(\s)+(\w)+(\s)+(\w)+(( )*\n)|$)')
re9 = re.compile('.*orwarded.*')
re10 = re.compile('From.*|Sent.*|cc.*|Subject.*|Embedded.*|http.*|\w+\.\w+|.*\d\d/\d\d/\d\d\d\d.*')
re11 = re.compile(' [\d:;,.]+ ')


from collections import defaultdict

docs = []
docs_num_dict = [] # Stores email sender's name and number

chdir('/home/peter/Downloads/enron')
# For each user we extract all the emails in their inbox

names = [i for i in listdir()]
m = 0
for name in names:
    sent = '/home/peter/Downloads/enron/' + str(name) + '/sent'   
    try: 
        chdir(sent)
        d = []
        for email in listdir():          
            text = open(email,'r').read()
            # Regular expressions are used below to remove 'clutter'
            text = re.sub(re0, ' ', text)
            text = re.sub(re1, ' ', text)
            text = re.sub(re2, ' ', text)
            text = re.sub(re3, ' ', text)
            text = re.sub(re4, ' ', text)
            text = re.sub(re5, ' ', text)
            text = re.sub(re6, ' ', text)
            text = re.sub(re7, ' ', text)
            text = re.sub(re8, ' ', text)
            text = re.sub(re9, ' ', text)
            text = re.sub(re10, ' ', text)
            text = re.sub(re11, ' ', text)
            docs.append(text)
            d.append(text)
        docs_num_dict.append((m,[name,d]))
        m += 1
    except:
        pass
    
docs_num_dict = dict(docs_num_dict)

# To build the dictionary
from collections import defaultdict
d = defaultdict(int)

# We now employ the techniques as outline in the second link at the top - see **
from stop_words import get_stop_words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

texts = []

for i in range(0,len(docs_num_dict.items())):
    new_docs_num_dict_1 = []
    for doc in docs_num_dict[i][1]:
        # Tokenization
        raw = doc.lower()
        tokens = tokenizer.tokenize(raw)

        # Removing stop words

        # create English stop words list
        en_stop = get_stop_words('en')

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]

        # Stemming 

        # Create wordnet_lemmatizer of class WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()

        # lemmatize token
        lemmatized_tokens = [wordnet_lemmatizer.lemmatize(i) for i in stopped_tokens]

        texts.append(lemmatized_tokens)
        new_docs_num_dict_1.append(lemmatized_tokens)

        # We now build the dictionary
        for word in lemmatized_tokens:
            d[word] += 1  
    docs_num_dict[i][1] = new_docs_num_dict_1

import json

chdir('/home/peter/Topic_Modelling/LDA/')

# Save the texts file as texts_raw (will be edited again below)
with open('texts_raw.jsn','w') as f:
    json.dump(texts,f)
f.close()

# Save the dictionary d
with open('d.jsn','w') as f:
    json.dump(d,f)
f.close()

import json

chdir('/home/peter/Topic_Modelling/LDA/')

# Loading the raw texts file
with open('texts_raw.jsn','r') as f:
    texts = json.load(f)
f.close()
    
# Loading the dictionary d 
with open('d.jsn','r') as f:
    d = json.load(f)
f.close()

from collections import defaultdict
docs_name_dict = []

for i in range(0,len(docs_num_dict.items())):
    temp_dict = defaultdict(int)
    for j in docs_num_dict[i][1]:
        for k in j:
            temp_dict[k] += 1
    # Append the temporary dictionary to docs_name_dict
    docs_name_dict.append((docs_num_dict[i][0],temp_dict)) 
docs_name_dict = dict(docs_name_dict)

num_docs = len(texts)
temp_texts = texts
texts= []
upper_lim = int(0.20*num_docs)

for doc in temp_texts:
    temp_doc = []
    for word in doc:
        # If the word is in the required interval, we add it to a NEW texts variable
        if 4 < d[word] < upper_lim and len(word) > 2:
            temp_doc.append(word)
        # If the word is not in the required interval, 
        # we lower the index of the word in the docs_name_dict dictinoary
        else:
            for group in docs_name_dict.items():
                person = group[0]
                if word in docs_name_dict[person]:
                    if docs_name_dict[person][word] > 1:
                        docs_name_dict[person][word] -= 1
                    else:
                        del docs_name_dict[person][word]
    texts.append(temp_doc)

import json
chdir('/home/peter/Topic_Modelling/LDA/')

# We save the new 'refined' texts file
with open('texts.jsn','w') as f:
    json.dump(texts,f)
f.close()

import pickle
chdir('/home/peter/Topic_Modelling/LDA/')

# We save the docs_name_dict global person, word-count dictionary
pickle.dump( docs_name_dict , open( "docs_name_dict.p", "wb" ) )

import json
chdir('/home/peter/Topic_Modelling/LDA/')

# Loading the texts file
with open('texts.jsn', 'r') as f:
    texts = json.load(f)
f.close()

import pickle
chdir('/home/peter/Topic_Modelling/LDA/')

# Loading the docs_name_dict dicitonary
docs_name_dict = pickle.load( open( "docs_name_dict.p", "rb" ) )

# Constructing a document-term matrix

from gensim import corpora, models

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

ldamodel = models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=350)

import json

chdir('/home/peter/Topic_Modelling/LDA/LDAdata_results')

# Saving the dictionary
dictionary.save('dictionary')

# Saving the corpus    
with open('corpus.jsn','w') as f:
    json.dump(corpus,f)    
f.close()

# Saving the ldamodel
ldamodel.save('ldamodel')

from gensim import corpora

chdir('/home/peter/Topic_Modelling/LDA/LDAdata_results')

# Load dictionary
dictionary = corpora.Dictionary.load('dictionary')

from gensim import models

chdir('/home/peter/Topic_Modelling/LDA/LDAdata_results')

# Load ldamodel
ldamodel = models.LdaModel.load('ldamodel') 

import json

chdir('/home/peter/Topic_Modelling/LDA/LDAdata_results')

# Load corpus
with open('corpus.jsn','r') as f:
    corpus = json.load(f)
f.close()

num_topics = 20
num_words = 10

List = ldamodel.print_topics(num_topics, num_words)
Topic_words =[]
for i in range(0,len(List)):
    word_list = re.sub(r'(.\....\*)|(\+ .\....\*)', '',List[i][1])
    temp = [word for word in word_list.split()]
    Topic_words.append(temp)
    print('Topic ' + str(i) + ': ' + '\n' + str(word_list))
    print('\n' + '-'*100 + '\n')

for i in range(0,len(Topic_words)):
    temp = Topic_words[i]
    sort_key = lambda s: (-len(s), s)
    temp.sort(key = sort_key)
    print(temp)
    Topic_words[i] = temp

import json

chdir('/home/peter/Topic_Modelling/LDA/LDAdata_results')

# Saving the list of words
with open('topic_words.jsn','w') as f:
    json.dump(Topic_words,f)
f.close()

import json

chdir('/home/peter/Topic_Modelling/LDA/LDAdata_results')

with open('topic_words.jsn','r') as f:
    Topic_words = json.load(f)
f.close()

import warnings
warnings.filterwarnings('ignore')

import pyLDAvis.gensim

lda_visualise = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(lda_visualise)

from palettable.tableau import Tableau_20

topic_colour_gen = []
for i in range(0,num_topics):
    topic_colour_gen.append((i, Tableau_20.hex_colors[i]))
    
topic_colours = dict(topic_colour_gen)

from nltk.stem.wordnet import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from collections import defaultdict
import re

doc = ''

def match_words(word):
    word_edit = word.lower()
    try:
        word_edit = tokenizer.tokenize(word_edit)[0]
    except:
        pass
    return wordnet_lemmatizer.lemmatize(word_edit)
    
def build_html_colour(word, topic):
    #return " <font color=" + str(topic_colours[topic]) + "'>" + word + "</font> "
    return ' <span style="background-color: ' + str(topic_colours[topic])  +'">' + word + '</span>'

def read_doc(doc):
    chdir('/home/peter/Topic_Modelling/LDA/text_files')
    doc = open(str(doc),'r').read()
    
    # Variables so recalculation is not necessary
    doc_split = doc.split()
    
    # Build dictionary of topic's distribution for a given document
    num_topics_weight = 0
    Topics = defaultdict(int)
    for word in doc_split:
        word_edit = match_words(word)
        try:
            word_topics = ldamodel.get_term_topics(word_edit)
            if word_topics:
                for topic in word_topics:
                    Topics[topic[0]] += topic[1]
                    num_topics_weight += topic[1]            
        except:
            pass
    # Find topic info
    # Append Topic, number of words in document from given topic and doc percentage of topic
    Topic_info = []
    for topic in Topics:
        Topic_info.append([topic, Topics[topic], round((Topics[topic]/num_topics_weight)*100)]) 
    
    # Topic info for three most prevalent topics for a given document
    Topic_info_top3 = []
    Topic_info_copy = []
    for i in Topic_info:
        Topic_info_copy.append(i)
    
    for i in range(0,3):
        max = Topic_info_copy[0]
        for topic in Topic_info_copy:
            if topic[2] > max[2]:
                max = topic
        Topic_info_top3.append(max)
        Topic_info_copy.remove(max)
        
    
    # Format the document according to topics
    for word in doc_split:
        word_edit = match_words(word)
        try:
            topic = ldamodel.get_term_topics(word_edit)[0][0]
            if (topic == Topic_info_top3[0][0]) or (topic == Topic_info_top3[1][0]) or (topic == Topic_info_top3[2][0]):
                doc = doc.replace( ' ' + word + '', build_html_colour(word,topic))
                #doc = doc.replace( '' + word + ' ', build_html_colour(word,topic))
        except:
            pass
    doc = re.sub(r'\n','<br>',doc)
    
    Output = []
    for item in Topic_info_top3:
        colour = build_html_colour('Topic ' + str(item[0]), item[0])
        topic_info = colour + ': ' + str(item[2]) + '% ' + str(Topic_words[item[0]])
        Output.append(topic_info)
    return Output, doc

# Example from http://jakevdp.github.io/blog/2013/06/01/ipython-notebook-javascript-python-communication/ adapted for IPython 2.0

#Input the document we want to read

doc = 'dickson-s_3.'

from IPython.display import HTML

input_form = """
<div style="background-color:white; border:solid black; width:1100px; padding:20px;">
<p>"""+read_doc(doc)[0][0]+"""</p>
<p>"""+read_doc(doc)[0][1]+"""</p>
<p>"""+read_doc(doc)[0][2]+"""</p>
<p>"""+read_doc(doc)[1]+"""</p>
</div>
"""

HTML(input_form) # + javascript)

from collections import defaultdict

def get_person_topics(person):
    person_topics = defaultdict(int)
    total = 0
    for word in docs_name_dict[person]:
        try:
            term_topics = ldamodel.get_term_topics(word)
            if term_topics:
                for topic_tuple in term_topics:
                    person_topics[topic_tuple[0]] += topic_tuple[1]
                    total += topic_tuple[1]
        except:
            pass
        
    #scale the values
    for person in person_topics:
        person_topics[person] = person_topics[person]/total
    return person_topics

def get_topic_persons(topic):
    specific_topic_persons = defaultdict(int)
    
    total = 0
    for person in docs_name_dict:
        person_topics = get_person_topics(person)
        person_value = person_topics[topic]
        specific_topic_persons[person] += person_value
        total += person_value
    
    
    #Scale the numbers in the dictionary to a percentage
    for person in docs_name_dict:
        specific_topic_persons[person] = specific_topic_persons[person]/total
        
    return specific_topic_persons
                

# Finding top person for a given topic

topic_person = get_topic_persons(10)
maximum_person = max(topic_person.keys(), key=(lambda key: topic_person[key]))
print(maximum_person, '{0:.2%}'.format(topic_person[maximum_person]))

# Finding top topic for a given person

person_topic = get_person_topics('allen-p')
maximum_topic = max(person_topic.keys(), key=(lambda key: person_topic[key]))
print(maximum_topic, '{0:.2%}'.format(person_topic[maximum_topic]))

def get_tot_words_person(person):
    n = 0
    for word in docs_name_dict[person]:
        n += docs_name_dict[person][word]
    return n

Data = []
list_of_names = []
list_of_names_dup = []
for name in docs_name_dict:
    list_of_names.append(name.capitalize().replace('-',', '))
    list_of_names_dup.append(name)
list_of_names.sort()
list_of_names_dup.sort()

for i in range(0,len(list_of_names)):
    name = list_of_names[i][0:-1]
    first_name = list_of_names[i][-1].capitalize()
    list_of_names[i] = name + first_name
    Data.append([name+first_name,list_of_names_dup[i],get_tot_words_person(list_of_names_dup[i])])

for data in Data:
    name = data[1]
    person_topics = get_person_topics(name)
    person_topics = [(v, k) for k, v in person_topics.items()]
    person_topics.sort()
    person_topics.reverse()
    for tuples in person_topics:
        data.append(tuples[1])
        data.append(tuples[0])
    L = range(0,20)
    for num in L:
        if num not in data:
            data.append(num)
            data.append(0)
    

Data = [['Employee', 'id', 'tot_words', 'A', 'Ap', 'B', 'Bp', 'C', 'Cp'
         , 'D', 'Dp', 'E', 'Ep', 'F', 'Fp', 'G', 'Gp', 'H', 'Hp',
         'I', 'Ip', 'J', 'Jp', 'K', 'Kp', 'L', 'Lp', 'M', 'Mp', 'N', 
         'Np', 'O', 'Op', 'P', 'Pp', 'Q', 'Qp', 'R', 'Rp', 'S', 'Sp', 'T', 'Tp']] + Data

import csv

with open("bubbles_data.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(Data)
f.close()



