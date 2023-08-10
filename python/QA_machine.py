# First step, load the relevant libraries and classes

from question_processing import Question
from synonyms import Synonyms
from Word2VecModel import W2V_Model 
from spanish_tagger import Spanish_Postagger
from query import Query
from index import go
from retrieval import ScoreParagraphs
from ngrams import retrieve_model
import json
import math

# Then name the different files you will use to make the code work

tagfile = 'stanford-postagger-full-2016-10-31/models/spanish.tagger'
jarfile = 'stanford-postagger-full-2016-10-31/stanford-postagger.jar'

# Data for synonyms API
APIfile = 'http://store.apicultur.com/api/sinonimosporpalabra/1.0.0/' 
token = 'f7JE_2svUVwP5ARGfw8aQhnLXlga'

# Data for w2v model
w2vfile = 'SBW-vectors-300-min5.txt'

# Data for question_type
question_json = 'Data/question_type.json'
stopwords = 'Data/stopwords.json'

query = Query(APIfile, token, tagfile, jarfile, 
    w2vfile)

question = "¿Cuál es el castigo por homicidio?" #Translation: What is the punishment for murder?

#set_question loads a question and a json file with info about the words that represent 
#a question type (i.e. time: years, period, days, etc )
query.set_question(question, question_json)

#get_query gives you a specific query list
query.get_query(stopwords)

print(query.query)
print(query.qtype)

query.add_words(["asesinato", "imprudencial"])

query.remove_words(["asesinato", "doloso"])

query.query

query.W2V.find_concepts(positive = ["rey", "mujer"], negative = ["hombre"], top_n = 1)

query.W2V.intruder(["rana", "pato", "simio", "pelota"])

go()

top_results = 20
words = list(query.query)
test = ScoreParagraphs(question, words, stem=True)
results = test.texts(top_results, method='bm25')

for i in results.text:
    print(i)
    print("--------------")

clf = retrieve_model()
clfs = clf.predict(results)
probs = [math.exp(i)-1 for i in clf.predict(results)]

print("Maybe these are your results...\n")
for i in range(len(results.text)):
    if probs[i] > 0:
        print("--------------")
        print(results.text[i])

# question = "¿Cuál es el castigo por extraer hidrocarburos ilegalmente?" 
# Translation: What is penalty for illegaly extracting hydrocarbons?

question = "¿Cuál es la multa por traición a la patria?" 
# Translation: What is penalty for not paying taxes?

query.set_question(question, question_json)
query.get_query(stopwords)
query.add_words(["multa", "sanción"])
words = list(query.query)
words

test = ScoreParagraphs(question, words, stem=True)
results = test.texts(top_results, method='bm25')

clf = retrieve_model()
clfs = clf.predict(results)
probs = [math.exp(i)-1 for i in clf.predict(results)]

print("Maybe these are your results...\n")
for i in range(len(results.text)):
    if probs[i] > 0:
        print("--------------")
        print(results.text[i])



