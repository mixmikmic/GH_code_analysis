#imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML

#constants
get_ipython().magic('matplotlib inline')
sns.set_style("dark")
sigLev = 3
figWidth = figHeight = 5

okCupidFrame = pd.read_csv("data/JSE_OkCupid/profiles.csv")

numObservations = okCupidFrame.shape[0]
numFeatures = okCupidFrame.shape[1]

#make numMissing for a given column
def numMissing(col):
    #helper that checks the number of observations missing from a given col
    missingRows = col[col.isnull()]
    return missingRows.shape[0]
#then apply over our feature set
missingSummaryFrame = okCupidFrame.apply(numMissing,axis = 0)
display(missingSummaryFrame)

#plot age
plt.hist(okCupidFrame["age"])
#then make some labels
plt.xlabel("Age (In Years)")
plt.ylabel("Count")
plt.title("Distribution of Age")

plt.hist(okCupidFrame["income"])
plt.xlabel("Income (In USD)")
plt.ylabel("Count")
plt.title("Distribution of Income")

numNotReportIncome = okCupidFrame[okCupidFrame["income"] == -1].shape[0]
propNotReportIncome = float(numNotReportIncome) / okCupidFrame.shape[0]
#get percent
percentMul = 100
percentNotReportIncome = propNotReportIncome * percentMul

filteredOkCupidFrame = okCupidFrame[okCupidFrame["essay0"].notnull()]

#language imports
import nltk
import collections as co
import StringIO
import re
#find full distribution of word frequencies
#write the all to a string wrtier
stringWriteTerm = StringIO.StringIO()
filteredOkCupidFrame["essay0"].apply(lambda x: stringWriteTerm.write(x))
#get the full string from the writer
summaryString = stringWriteTerm.getvalue()
stringWriteTerm.close()
#lower and split into series of words (tokens) on multiple split criteria
summaryString = summaryString.lower()
#split on ".", " ", ";", "-", or new line
summaryWordList = re.split("\.| |,|;|-|\n",summaryString)

#keep only legal words, and non stop-words (i.e. "." or "&")
#get counter of legal English
legalWordCounter = co.Counter(nltk.corpus.words.words())
stopWordsCounter = co.Counter(nltk.corpus.stopwords.words())
#filter narrativeWordList
filterSummaryWordList = [i for i in summaryWordList
                           if i in legalWordCounter and
                              i not in stopWordsCounter]
#counter for the legal words in our filtered list
filteredWordCounter = co.Counter(filterSummaryWordList)

#make series of word frequency ordered by most common words
wordFrequencyFrame = pd.DataFrame(filteredWordCounter.most_common(),
                                  columns = ["Word","Frequency"])
wordFrequencyFrame["Density"] = (wordFrequencyFrame["Frequency"] /
                                np.sum(wordFrequencyFrame["Frequency"]))
#then plot rank-density plot
#for the sake of easier visuals, we will log the rank
desiredLineWidth = 3
plt.plot(np.log(wordFrequencyFrame.index+1),wordFrequencyFrame["Density"],
         lw = desiredLineWidth)
plt.xlabel("Log-Rank")
plt.ylabel("Density")
plt.title("Log(Rank)-Density Plot\nFor Words in our Summary Set")

topLev = 10
topTenWordFrame = wordFrequencyFrame.iloc[0:topLev,:].loc[:,
                                                        ["Word","Frequency"]]
#then display
display(HTML(topTenWordFrame.to_html(index = False)))

#import our count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
#make a vocab dictionary
counterList = filteredWordCounter.most_common()
vocabDict = {}
for i in xrange(len(counterList)):
    rankWord = counterList[i][0]
    vocabDict[rankWord] = i
#initialize vectorizer
vectorizer = CountVectorizer(min_df=1,stop_words=stopWordsCounter,
                             vocabulary = vocabDict)
#then fit and transform our summaries
bagOfWordsMatrix = vectorizer.fit_transform(filteredOkCupidFrame["essay0"])

#get language frame
langFrame = pd.DataFrame(bagOfWordsMatrix.toarray(),
                         columns = vectorizer.get_feature_names())
#import linear model
import sklearn.linear_model as lm
#build model
initialLinearMod = lm.LinearRegression()
initialLinearMod.fit(langFrame,filteredOkCupidFrame["age"])

#get predictions
predictionVec = initialLinearMod.predict(langFrame)

