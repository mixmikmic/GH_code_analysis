import string
import re
import os
import tempfile
import logging
from gensim import corpora
from gensim import models
from gensim.corpora import Dictionary
numberOfTopics = 150

table = open("../data/paperTable.tsv","r")
entries = []
for line in table:
    entries.append(line.split('\t'))
table.close()

# Read in abstract year of publication, title of abstract, and abstract text
abstracts = []
titleOfAbstract = []
yearOfAbstract = []
for articles in entries:
    titleOfAbstract.append(articles[0])
    abstracts.append(articles[1]+articles[2]+articles[3])
    yearOfAbstract.append(articles[4][:-1])

# Create a set of frequent words
stopFile = open("../data/stopwords.txt","r")
stopWords = stopFile.read().splitlines()
stopWords.append("\xc2\xa9") #This is the copyright symbol, this shows up in every abstract and should not be apart of the corpus
stopWords.extend(["\u2019","\u03bc","bee","bees","honey","honeybee","honeybees"])
stopList = set(stopWords)

# Lowercase each document, split it by white space and filter out stopWords
texts = []
for document in abstracts:
    docwords = []
    for word in document.lower().split():
        word = re.sub(r'[^\w\s]','',word)
        word = re.sub(r'\.+$','',word)
        isNumber = re.compile('^[0-9]+$')
        if isNumber.search(word):
            word = ''
        if word not in stopList and word!='':
            docwords.append(word)
    texts.append(docwords)

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
        
processedCorpus = [[token for token in text if frequency[token] > 5] for text in texts]

# Save the dictionary of tokens
tempFolder = tempfile.gettempdir()
dictionary = corpora.Dictionary(processedCorpus)
dictionary.save(os.path.join(tempFolder,'words.dict'))

# Create general corpus and serialize in order for it to be iterated over
corpus = [dictionary.doc2bow(text) for text in processedCorpus]
corpora.MmCorpus.serialize(os.path.join(tempFolder, 'words.dict'), corpus)

# Train the model and set number of topics
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
lda = models.ldamodel.LdaModel(corpus,id2word=dictionary,num_topics=numberOfTopics)

#Sorts the word topics in decending order based on their greatest phi value

temp = phiValues
for x in xrange(0,len(temp)):
    temp[x][1].sort(key=lambda x:x[1],reverse=True)
temp.sort(key=lambda x:x[1][0][1],reverse=True)

# Sort the most interesting words per topic per document
# This cell does not need to be run if only trying to create Top Nine terms per paper
topicOrganizingFile = open("../data/topicorganization.tsv","w")
for x in xrange(0,len(abstracts)):
    doc = dictionary.doc2bow(abstracts[x].split())
    docTopics, wordTopics, phiValues = lda.get_document_topics(doc, per_word_topics=True)
    topicOrganizingFile.write(yearOfAbstract[x]+"\t"+titleOfAbstract[x]+"\t")
    for y in xrange(0,min(3,len(docTopics))):
        topicnumber = docTopics[y][0]
        topicOrganizingFile.write(str(lda.show_topic(topicnumber))+"\t")
        for z in xrange(0,len(phiValues)):
            phiValues[z][1].sort(key=lambda q:q[1],reverse=True)
        phiValues.sort(key=lambda q:q[1][0][1],reverse=True)
        curindex=0
        topwords = ""
        for z in xrange(0,3):
            while curindex<len(phiValues) and phiValues[curindex][1][0][0]!=topicnumber:
                curindex+=1
            if(curindex>=len(phiValues)):break
            print len(phiValues)
            print dictionary[phiValues[curindex][0]]
            topwords+=str(dictionary[phiValues[curindex][0]].encode('utf-8').strip())+" "
            curindex+=1
        filter(lambda a:a[0]!=topicnumber,phiValues)
        topicOrganizingFile.write(topwords+"\t")
    topicOrganizingFile.write("\n")
topicOrganizingFile.close()

        

#Makes the top nine terms for each document

topNineFile = open("../data/Docbow/TopNineTerms-"+str(numberOfTopics)+".tsv","w")
for x in xrange(0,len(abstracts)):
    doc = dictionary.doc2bow(abstracts[x].split()) # Convert to bag of words format first
    # Get the topics and words associated with each document
    docTopics, wordTopics, phiValues = lda.get_document_topics(doc, per_word_topics=True)
    topNineFile.write(yearOfAbstract[x]+"\t"+titleOfAbstract[x]+"\t")
    for z in xrange(0,len(phiValues)):
        phiValues[z][1].sort(key=lambda q:q[1],reverse=True)
    phiValues.sort(key=lambda q:q[1][0][1],reverse=True)
    nineWords = ""
    for x in phiValues[:9]:
        nineWords+= dictionary[x[0]] + " "
    topNineFile.write(nineWords.encode('utf-8')+"\n")



