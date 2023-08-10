get_ipython().magic('pylab inline')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import re
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import wordnet as wn
nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}

import warnings
warnings.filterwarnings('ignore')

from pylab import rcParams
rcParams['figure.figsize'] = 20,10

data_train = pd.read_json('../Dataset/Random Acts Of Pizza/train.json')
data_train['data_type'] = 'train'
y = data_train.pop('requester_received_pizza')
data_train.head(2)

data_test = pd.read_json('../Dataset/Random Acts Of Pizza/test.json')
data_test['data_type'] = 'test'
data_test.head(2)

not_present = []
for i in data_train.columns:
    if i not in data_test.columns:
        not_present.append(i)
data_train.drop(labels=not_present,axis=1,inplace=True)

## Combining the training and testing data
data = pd.concat([data_train,data_test],ignore_index=True)
data_copy = data.copy()
data.shape

data.head(2)

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

n_samples = 2000
n_features = 1000
n_topics = 20
n_top_words = 20

def applyNMF(data_samples):
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0,min_df=1,stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    print("Fitting the NMF model with tf-idf features,"
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
    return nmf.transform(tfidf)

topics = applyNMF(data['request_text_edit_aware'])
print(topics.shape)

topics_vec = np.argmax(topics,axis=1)

data['topics'] = topics_vec
data['topics'].head()

## Finding the nature of the topics generated
from collections import Counter
imp_topics = Counter(topics_vec).most_common(10)
print imp_topics

def find_topic(topic,remove_verbs=True):
    requests = data_copy[data['topics'] == imp_topics[topic][0]]['request_text_edit_aware']
    chain_requests = ''
    for request in requests:
        chain_requests += ('. '+request)
    chain_requests = re.sub('^[a-zA-Z]',' ',chain_requests)
    words = [word for word in chain_requests.split() if word not in stopwords.words("english")]
    if remove_verbs:
        words = [word for word in words if word in nouns]
    return Counter(words).most_common(100)

topic_words = []
for i in range(len(imp_topics)):
    words = find_topic(i)
    words = ' '.join([word[0] for word in words])
    topic_words.append(words)

train = data[data['data_type'] == 'train']
train.head(2)

train['received'] = y
train.head(2)

topic_df = []
for i in range(len(imp_topics)):
    topic_df.append([imp_topics[i][0],topic_words[i],100*float(train[train['topics'] == imp_topics[i][0]]['received'].sum())                     /len(train[train['topics'] == imp_topics[i][0]]['received'])])

topic_df = pd.DataFrame(topic_df,columns = ['Topic','Words','Success Rate'])
topic_df

topic_df.plot(kind='bar',y='Success Rate',x='Topic')
plt.xlabel('Topic')
plt.ylabel('Success Rate')
plt.title('Success Rate vs Topics')
plt.show()

train.dropna(inplace=True,axis=0)
train.tail(1)

train['request_length'] = [len(x.split()) for x in train['request_text_edit_aware']]
train.head(2)

length = []
def length_success(topic):    
    max_length = train[train['topics'] == topic]['request_length'].max()
    min_length = train[train['topics'] == topic]['request_length'].min()
    bin_size = (max_length - min_length)/20
    df = train[train['topics'] == topic]
    for i in range(10):
        df_one = df[(df['request_length'] >= min_length) & (df['request_length'] < min_length+bin_size)]
        df_new = df_one[df_one['received'] == True]
        if(len(df_one) == 0):
            df_one = ['a']
        length.append([topic,min_length,min_length+bin_size,float(len(df_new))/len(df_one)])
        min_length = min_length + bin_size

for topic in imp_topics:
    print 'Calculating length probabilities for {} topic..'.format(topic[0])
    length_success(topic[0])

df_length = pd.DataFrame(length,columns=['Topic','Lower Bound','Upper Bound','Probability Success'])
df_length.head(5)

df_length.to_csv('LengthCorrelation.csv',sep=',',columns=df_length.columns)

topic_points = []
for topic in imp_topics:
    points = []
    df_new = df_length[df_length['Topic'] == topic[0]]
    for i in range(8):    
        points.append(((df_new.iloc[i,1] + df_new.iloc[i,2]/2),df_new.iloc[i,3]))
    topic_points.append(points)

i = 1
for points in topic_points:
    plt.subplot(3,4,i)
    plt.plot([point[0] for point in points],[point[1] for point in points])
    plt.ylabel('Probability of Success')
    plt.title('Topic {}'.format(imp_topics[i-1][0]))
    i += 1
    if i > 10:
        i = 1
plt.show()

import re
regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

evidentiality,i = np.ones((len(train['request_text_edit_aware']))),0
for request in train["request_text_edit_aware"]:
    url = re.findall(regex,request)
    if len(url) <= 0:
        evidentiality[i] = 0
    i += 1

train['evidentiality'] = evidentiality
train.head(2)

## How evidentiality affects the success rate
total = train[train['evidentiality'] == 1].received
success = len(total[total == True])
print 'Percentage of successful requests with evidence: {}%'.format(round(float(success)*100/len(total),3))

total = train[train['evidentiality'] == 0].received
success = len(total[total == True])
print 'Percentage of successful requests without evidence: {}%'.format(round(float(success)*100/len(total),3))

evidence_relation = pd.Series({'Success with evidence':35.192,'Success without evidence':23.794})
evidence_relation.to_csv('evidenceRelation.csv',sep=',')
evidence_relation.plot(kind='bar',rot=0)
plt.ylabel('Percentage of Successful request')
plt.title('How evidence effects a successful request')
plt.show()

reciprocity,i = np.zeros((len(train['request_text_edit_aware']),)),0
regex = 'return the favor|pay it forward|pay it back'
for request in train['request_text_edit_aware']:
    match = re.search(regex,request)
    if match:
        reciprocity[i] = 1
    i += 1

train['reciprocity'] = reciprocity
train.head(2)

## Finding percentage of successful request with reciprocity and without it
total = train[train['reciprocity'] == 1].received
success = len(total[total == True])
print 'Percentage of successful requests with reciprocity: {}%'.format(round(float(success)*100/len(total),3))

total = train[train['reciprocity'] == 0].received
success = len(total[total == True])
print 'Percentage of successful requests with reciprocity: {}%'.format(round(float(success)*100/len(total),3))

reciprocity_relation = pd.Series({'Success with reciprocity':30.058,'Success without reciprocity':23.8})
reciprocity_relation.to_csv('reciprocity_relation.csv',sep=',')
reciprocity_relation.plot(kind='bar',rot=0)
plt.ylabel('Percentage of Successful request')
plt.title('How reciprocity effects a successful request')
plt.show()

train.head(2)

narrative = {'Money': 'money now broke week until time last day when today tonight paid next first night after tomorrow month while account before long Friday rent buy bank still bills bills ago cash due due soon past never paycheck check spent years poor till yesterday morning dollars financial hour bill evening credit budget loan bucks deposit dollar current payed'.split(),'Job':'work job paycheck unemploymentinterview fired employment hired hire'.split(),'Student':'collegestudent school roommate studying university finals semester classstudy project dorm tuition'.split(),'Family':'family mom wife parentsmother hus- band dad son daughter father parent mum'.split(),'Craving':'friend girlfriend craving birthday boyfriend celebrate party game games moviedate drunk beer celebrating invited drinks crave wasted invite'.split()}

request_narrative = []
narration = []
for request in train['request_text_edit_aware']:
    word_count = {'Money':0,'Job':0,'Student':0,'Family':0,'Craving':0}
    n = 0
    for word in request.split():
        for lexicon in narrative:
            if word in narrative[lexicon]:
                word_count[lexicon] += 1
    for lexicon in word_count:
        n += word_count[lexicon]
    request_narrative.append(word_count)
    try:
        narration.append(float(n)/len(request.split()))
    except:
        narration.append(0)

train['narrative'] = narration

train.head(2)

from nltk.parse.stanford import StanfordDependencyParser
import string

path_to_jar = '../../../Downloads/stanford-parser-full-2014-08-27/stanford-parser.jar'
path_to_models_jar = '../../../Downloads/stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models.jar'

dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

def dep_parse(phrase):
    words = [word for word in set(word_tokenize(phrase)) if word not in string.punctuation]
    result = dependency_parser.raw_parse(phrase)
    dep = result.next()
    if dep == None:
        return ''
    triplet = list(dep.triples())
    if triplet == None:
        return ''
    parse = []
    for i in triplet:
        try:
            parse.append("{}({}-{}, {}-{})".format(i[1],i[0][0],words.index(i[0][0])+1,i[2][0],words.index(i[2][0])+1))
        except:
            pass
    return parse

from nltk.tokenize import word_tokenize

import nltk.data
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

## Warning : TAKES A LONG TIME TO RUN
## DON'T RUN UNTIL REQUIRED
text_documents,i = [],2501
for text in train:
    i += 1
    parsed_sents = {}
    try:
        parsed_sents['text'] = text
        parsed_sents['sentences'] = [sents for sents in np.asarray(tokenizer.tokenize(text)) if len(sents.split()) > 1]
        temp = []
        for sentence in parsed_sents['sentences']:
            try:
                temp.append(dep_parse(sentence))
            except:
                pass
        parsed_sents['parses'] = temp
    except:
        print text
        break
    text_documents.append(parsed_sents)
    print '{} requests parsed...'.format(i)
    if i%100 == 0:
        get_ipython().magic('store text_documents >> test_documents_new.py')

filename = str(raw_input('Enter the filename: '))
get_ipython().magic('store text_documents >> filename')

def statistical_sim(sent1, sent2):
    '''
    Statistical similarity between sentences
    based on the cosine method
    Returns: float (the cosine similarity b/w sent1 and sent2)
    '''
    sent_token1 = Counter(sent1)
    sent_token2 = Counter(sent2)

    intxn = set(sent_token1) & set(sent_token2)
    numerator = sum([sent_token1[x] * sent_token2[x] for x in intxn])

    mod1 = sum([sent_token1[x]**2 for x in sent_token1.keys()])
    mod2 = sum([sent_token2[x]**2 for x in sent_token2.keys()])
    denominator = sqrt(mod1)*sqrt(mod2)

    if not denominator:
        return 0.0

    return float(numerator)/denominator

## Sanity check for statistical similarity
sent1 = 'Hello my name is Najeeb Khan'
sent2 = 'Hello my name is Najeeb Khan'
statistical_sim(sent1,sent2)

## Warning : Takes a long time to RUN
## Do not RUN until required
i = 0
similarity = []
for request1 in data_train['request_text_edit_aware']:
    cosine_sim = []
    for request2 in data_train['request_text_edit_aware']:
        if request1 != request2:
            cosine_sim.append(statistical_sim(request1,request2))
    similarity.append([np.argmax(np.asarray(cosine_sim)),np.max(np.asarray(cosine_sim))])
    i += 1
    if i%100 == 0:
        get_ipython().magic('store similarity >> similarity.py')
        print 'Finding similarity in request {}'.format(i)

train['similarity'] = pd.read_json('../../Dataset/Random Acts Of Pizza/data_train.json')['similarity']

politeness_data = pd.read_csv('../../Dataset/Random Acts Of Pizza/politeness_one.csv',index_col=0)
part_two = pd.read_csv('../../Dataset/Random Acts Of Pizza/politeness_two.csv',index_col=0)

## One data is missing... So appending missing data
politeness_data = politeness_data.append({'text':data_train.iloc[2500,2],'polite':0.5,'impolite':0.5},ignore_index=True)

## Sanity Check for the size of dataset
print politeness_data.shape[0] + part_two.shape[0] == train.shape[0]

## Adding the politeness data into 'master' dataframe
politeness_data = politeness_data.append(part_two,ignore_index=True)
train['polite'] = politeness_data['polite']
train['impolite'] = politeness_data['impolite']

train.head(2)

train.to_json('../../Dataset/Random Acts Of Pizza/trainingData.json',orient='columns')

if train.isnull().values.any() == False:
    print 'Huzzah... No NaNs.. Mission Accomplished :)'



