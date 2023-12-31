docA = "the cat sat on my face"
docB = "the dog sat on my bed"

bowA = docA.split(" ")
bowB = docB.split(" ")

bowB

wordSet= set(bowA).union(set(bowB))

#all words in all bags/documents
wordSet

#I'll create dictionaries to keep my word counts.
wordDictA = dict.fromkeys(wordSet, 0)
wordDictB = dict.fromkeys(wordSet, 0)

#This is what one of them looks like
wordDictA

#now I'll count the words in my bags.
for word in bowA:
    wordDictA[word]+=1

for word in bowB:
    wordDictB[word]+=1

wordDictA

#Lastly I'll stick those into a matrix.
import pandas as pd
pd.DataFrame([wordDictA, wordDictB])

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bowCount)
    return tfDict

tfBowA = computeTF(wordDictA, bowA)
tfBowB = computeTF(wordDictB, bowB)

def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)
    
    #counts the number of documents that contain a word w
    idfDict = dict.fromkeys(docList[0].keys(),0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] +=1
                
    #divide N by denominator above, take the log of that
    for word, val in idfDict.items():
        idfDict[word]= math.log(N / float(val)) 

    return idfDict
    
   
    

idfs = computeIDF([wordDictA, wordDictB])

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidfBowA =  computeTFIDF(tfBowA, idfs)
tfidfBowB = computeTFIDF(tfBowB, idfs)

#Lastly I'll stick those into a matrix.
import pandas as pd
pd.DataFrame([tfidfBowA, tfidfBowB])



