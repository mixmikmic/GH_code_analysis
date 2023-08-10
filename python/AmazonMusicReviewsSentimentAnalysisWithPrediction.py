import pandas as pd

data = pd.read_csv("~/Desktop/cleanresult.csv", header=0, delimiter=",")

data['lsentiment']=0
print(data.shape)

#WordNet constructs are lemmas and synset
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
#SentiWordNet is a lexical resource for opinion mining.
#SentiWordNet assigns to each synset of WordNet three
#sentiment scores: positivity, negativity, and objectivity
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag

lemmatizer = WordNetLemmatizer()

# method for part of speech (POS) tagging
def penn_to_wn(tag):
     if tag.startswith('J'):
        return wn.ADJ
     elif tag.startswith('N'):
        return wn.NOUN
     elif tag.startswith('R'):
        return wn.ADV
     elif tag.startswith('V'):
        return wn.VERB
     return None

#method to find the sentiment.
#1. first word is tokenized
#2. checked if the tokenized word is part of speech
#3. checked if a word belongs to lemma (root form) using lemmatizer
#4. checked if a word is making some meaning using sysnset and pos tagging
#if all the above criteria is met, we take sentiment of the word from synset using sentiWordNet
def swn_polarity(text):
    sentiment = 0
    tokens_count = 0
    #tokenizing
    for raw_sentence in text:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
        #check if word is in pos
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
            #Check if its  a lemma
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            #check if its in synsets
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 
            
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            #sentiment calculating
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
            
            if not tokens_count:
                return 0
            if sentiment >= 0:
                return 1
            

            return 0

#taking users review data for analysis
textdata1 = data["reviewText"]

#calling method to calc sentimemt
lsentiment=[]

for i in textdata1:
    lsentiment.append(swn_polarity(i))

resultsentiment=zip(textdata1,lsentiment)

#adding sentiment result to dataset
data['lsentiment']=pd.Series(lsentiment)
#output data to csv for analysis
#review.to_csv('output_SentimentResult.csv')
print "done"

#Prediction : Creating model for sentiment analysis using overall rating values in dataset

#Run a logistic regression
import statsmodels.api as sm2

#creating training set and test set and removing nan values usimg 80:20 split
train = data[:90000] 
test = data[90000:] 
train = train.fillna(0)
test=test.fillna(0)

#make prediction model using overall rating as independent variable ,predict sentiment

# Run a logistic regression 
import statsmodels.api as sm2
logit=sm2.Logit(train['lsentiment'].dropna(),train['overall'].dropna())
result=logit.fit() 
print(result.summary())

#prediction of sentiment based on confidence
predictSentiment = result.predict(test['overall'])

predictSentiment = (predictSentiment > 0.50).astype(int)
print(predictSentiment)

print("Confusion Matrix")
from sklearn.metrics import confusion_matrix

print(confusion_matrix(test['lsentiment'],predictSentiment))

#Accuracy of setiment analysis 
from sklearn.metrics import accuracy_score
print(accuracy_score(test['lsentiment'],predictSentiment))

























