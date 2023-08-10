import nltk
import json
file_data=""
reviews_final_str=""
reviews_str=""
reviews_neg_str=""
with open('D:/reviews1_1000_data.txt','r',encoding="utf8",newline='') as f:
    file_data = json.load(f)

'''print sample data of first review'''
print(file_data[0][0])

for r in file_data:
    for s in r:
        reviews_final_str = reviews_final_str + str(s['review_text'])
        if ((s['review_rating'][0][:1]=="1") or (s['review_rating'][0][:1]=="2")):
            reviews_neg_str = reviews_neg_str + str(s['review_text'])
        else:
            reviews_str = reviews_final_str + str(s['review_text'])
            
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktLanguageVars
'''We customize the ReviewLangVars class to separate sentences based on some additional keywords'''
class ReviewLangVars(PunktLanguageVars):
    sent_end_chars = ('pros:', 'cons:', '[','][','.','?','!')
    
    
sent_tokenizer1 = PunktSentenceTokenizer(lang_vars = ReviewLangVars())
sent_fullreview = sent_tokenizer1.tokenize(reviews_final_str)
sent_neg_review = sent_tokenizer1.tokenize(reviews_neg_str)
sent_review = sent_tokenizer1.tokenize(reviews_str)
sent_fullreview[:5]

import sys

from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
        """calculates the support for items in the itemSet and returns a subset
       of the itemSet each of whose elements satisfies the minimum support"""
        _itemSet = set()
        localSet = defaultdict(int)

        for item in itemSet:
                for transaction in transactionList:
                        if item.issubset(transaction):
                                freqSet[item] += 1
                                localSet[item] += 1

        for item, count in localSet.items():
                support = float(count)/len(transactionList)

                if support >= minSupport:
                        _itemSet.add(item)

        return _itemSet


def joinSet(itemSet, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))              # Generate 1-itemSets
    return itemSet, transactionList


def runApriori(data_iter, minSupport, minConfidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    oneCSet = returnItemsWithMinSupport(itemSet,
                                        transactionList,
                                        minSupport,
                                        freqSet)

    currentLSet = oneCSet
    k = 2
    while(currentLSet != set([])):
        largeSet[k-1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(currentLSet,
                                                transactionList,
                                                minSupport,
                                                freqSet)
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
            """local function which Returns the support of an item"""
            return float(freqSet[item])/len(transactionList)

    toRetItems = []
    for key, value in list(largeSet.items()):
        toRetItems.extend([(tuple(item), getSupport(item))
                           for item in value])

    toRetRules = []
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item)/getSupport(element)
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)),
                                           confidence))
    return toRetItems, toRetRules


def printResults(items):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    for item, support in sorted(items, key=lambda item_support: item_support[1], reverse=True):
        print(str(item), support)
    #print ("\n------------------------ RULES:")
    #for rule, confidence in sorted(rules, key=lambda rule_confidence: rule_confidence[1]):
        #pre, post = rule
        #print (str(pre), str(post), confidence)

import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

stemmer = nltk.stem.porter.PorterStemmer()

def stem(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    word = word.replace("'","").replace('"','').replace('.','')
    word1 = stemmer.stem(word)
    return word1

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool((2 <= len(word) <= 40) and word.lower() not in stopwords)
    return accepted
        
def get_terms(tree):
    term = [ stem(w) for w in tree if acceptable_word(w) ]
    yield term

def is_noun(n):
    if n=='NN' or n=='NNS' or n=='NNP' or n=='NNPS':
        return True

revset=[]
for line in sent_fullreview:
    a = nltk.word_tokenize(line)
    nouns = [word for (word, pos) in nltk.pos_tag(a) if is_noun(pos)] 
    terms = get_terms(nouns)
        
    for term in terms:   
        tempset=[]
        for word in term:
            tempset.append(word)
        revset.append(tempset)
print(revset[:20])

items, rules = runApriori(revset, 0.01, 0.05)
printResults(items)

def custom_liu_hu_lexicon(sentence):
    '''Takes in a sentence and returns the sentiment of the sentence by counting the no of positive and negitive 
    and negitive words and by reversing the sentiment if the words NO or NOT are present
    '''
    from nltk.corpus import opinion_lexicon
    from nltk.tokenize import treebank

    tokenizer = treebank.TreebankWordTokenizer()
    pos_words = 0
    neg_words = 0
    tokenized_sent = [word.lower() for word in tokenizer.tokenize(sentence)]

    x = list(range(len(tokenized_sent))) 
    y = []
    isNegation = False
    negationWords = ['no','not','never','none','hardly','rarely','scarcely','']

    for word in tokenized_sent:
        if word in opinion_lexicon.positive():
            pos_words += 1
            y.append(1) # positive
        elif word in opinion_lexicon.negative():
            neg_words += 1
            y.append(-1) # negative
        else:
            y.append(0) # neutral
            
        if word in negationWords:
            isNegation = True

    if pos_words > neg_words and isNegation==True:
        return 'neg'
    elif pos_words > neg_words:
        return 'pos'
    elif pos_words < neg_words and isNegation==True:
        return 'pos'
    elif pos_words < neg_words:
        return 'neg'
    elif pos_words == neg_words:
        return 'neutral'

neutral_review=[]
positive_review=[]
negative_review=[]
for sentence in sent_review:
    for i in items:
        if i[0][0] in sentence:
            #print(i[0][0] +"--" + sentence)
            x=custom_liu_hu_lexicon(sentence)
            if(x=="pos"):
                positive_review.append(sentence)
            elif(x=="neg"):
                negative_review.append(sentence)
            else:
                neutral_review.append(sentence)
            break

for sentence in sent_neg_review:
    for i in items:
        if i[0][0] in sentence:
            #print(i[0][0] +"--" + sentence)
            negative_review.append(sentence)
            break
print('done')            

print(positive_review[:10])

import pickle
with open('D:/all_pos_rating_op5.txt', 'wb') as fp:  
    pickle.dump(positive_review, fp)

print('done')    
    
with open('D:/all_neg_rating_op5.txt', 'wb') as fp:  
    pickle.dump(negative_review, fp)

print('done')

with open('D:/all_neutral_rating_op5.txt', 'wb') as fp:  
    pickle.dump(neutral_review, fp)
print('done')

import pickle
pos_sentences=[];neg_sentences=[];neutral_sentences=[]
with open ('D:/all_pos_rating_op5.txt', 'rb') as fp:
    pos_sentences = pickle.load(fp)
with open ('D:/all_neg_rating_op5.txt', 'rb') as fp:
    neg_sentences = pickle.load(fp)
with open ('D:/all_neutral_rating_op5.txt', 'rb') as fp:
    neutral_sentences = pickle.load(fp)

print(str(pos_sentences[:5]))

import nltk
pos_tokens=[];pos_tokens_postagged=[];neg_tokens=[];neg_tokens_postagged=[];neut_tokens=[];neut_tokens_postagged=[];
'''Tokenize the sentences into words'''
for sent in pos_sentences: 
    pos_tokens.append(nltk.word_tokenize(sent))    
for sent in neg_sentences:
    neg_tokens.append(nltk.word_tokenize(sent))    
for sent in neutral_sentences:
    neut_tokens.append(nltk.word_tokenize(sent))

'''Apply Part of speech tagging for the tokenized words'''
for sent in pos_tokens:    
    pos_tokens_postagged.append(nltk.tag.pos_tag(sent))
for sent in neg_tokens:    
    neg_tokens_postagged.append(nltk.tag.pos_tag(sent))
for sent in neut_tokens:    
    neut_tokens_postagged.append(nltk.tag.pos_tag(sent))  
    
print(pos_tokens_postagged[:5])

from nltk.corpus import stopwords
stopwords = stopwords.words('english')

lem = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
lem_word_mapping={}

gram = r"""       
    P1:{<JJ><NN|NNS>}
    P2:{<JJ><NN|NNS><NN|NNS>}
    P3:{<RB|RBR|RBS><JJ>}
    P4:{<RB|RBR|RBS><JJ|RB|RBR|RBS><NN|NNS>}
    P5:{<RB|RBR|RBS><VBN|VBD>}
    P6:{<RB|RBR|RBS><RB|RBR|RBS><JJ>}
    P7:{<VBN|VBD><NN|NNS>}
    P8:{<VBN|VBD><RB|RBR|RBS>}
"""

def leaves(tree):
    """Finds leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label() in ['P1','P2','P3','P4','P5','P6','P7','P8']):
        yield subtree.leaves()

def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    word1 = stemmer.stem(word)
    word1 = lem.lemmatize(word1)    
    if word!=word1:
        lem_word_mapping[word1]=word
    return word1

def get_terms(tree):
    """Returns the words after checking acceptable conditions, normalizing and lemmatizing"""
    term = [ normalise(w) for w in tree if acceptable_word(w) ]
    yield term
    
def get_t_norm(tree):
    """Parse leaves in chunk and return after checking acceptable conditions, normalizing and lemmatizing"""
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
        yield term

def extractOpinionPhrases(posTaggedData):
    '''Extract noun phrases from part of speech tagged tokenized words'''
    output=[]
    for tup in posTaggedData:
        chunk = nltk.RegexpParser(gram)
        tr = chunk.parse(tup)
        term = get_t_norm(tr)

        for ter in term:    
            wordConcat=""
            for word in ter:
                if wordConcat=="":
                    #Replace good, wonderful and awesome with great
                    wordConcat = wordConcat + word.replace("good","great").replace("wonderful","great").replace("awesome","great").replace("awesom","great")
                else:
                    wordConcat = wordConcat + " " +  word
            if(len(ter)>1):
                output.append(wordConcat) 
    return output

ExtractedWords_pos = extractOpinionPhrases(pos_tokens_postagged)
ExtractedWords_neg = extractOpinionPhrases(neg_tokens_postagged)
        
print(ExtractedWords_pos[:50])

freqdist_neg = nltk.FreqDist(word for word in ExtractedWords_neg)
mc_neg = freqdist_neg.most_common()
freqdist_pos = nltk.FreqDist(word for word in ExtractedWords_pos)
mc_pos = freqdist_pos.most_common()

print(mc_pos[:50])
print("----")
print(mc_neg[:50])

import inflect
p = inflect.engine()
def replacewords(mc):
    newmc=[]
    for a in mc:
        newword="";found=False;
        for b in a[0].split():            
            for x in lem_word_mapping:
                #print(x)
                #print(b)
                if b==x:
                    found=True
                    sing=(lem_word_mapping[x] if p.singular_noun(lem_word_mapping[x])==False else p.singular_noun(lem_word_mapping[x]))
                    if newword=="":
                        newword = newword + sing
                    else:
                        newword = newword + " " +  sing
            if found==False:
                if newword=="":
                    newword = newword + b
                else:
                    newword = newword + " " +  b
                    #print(newword)
        newmc.append((newword,a[1]))
    return newmc

final_neg = replacewords(mc_neg)
final_pos = replacewords(mc_pos)

print("Postive Opinon phrases:")
print(final_pos[:50])

print("Negitive Opinion phrases:")
print(final_neg[:50])

def finResult(itemArr, opinionPhrases, sentenceArr):
    for item,support in sorted(items, key=lambda item_support: item_support[1], reverse=True):
        count=0
        print("----------"+item[0]+"----------")
        for phrase,freq in sorted(opinionPhrases, key=lambda phrase_freq: phrase_freq[1], reverse=True): 
            pcount=0
            if normalise(item[0]) in normalise(phrase):
                count+=1
                print("---"+phrase+"---")
                for l in sentenceArr:
                    if normalise(phrase) in normalise(l):   
                        for b in zip(l.split(" ")[:-1], l.split(" ")[1:]):
                            #print(b[0]+" "+b[1])
                            if normalise(b[0])==normalise(item[0]):
                                print(l.replace("'","").replace("]","").replace("[",""))
                                pcount+=1
                                break
                            elif (normalise(b[0])+" "+normalise(b[1]))==normalise(item[0]):
                                print(l.replace("'","").replace("]","").replace("[",""))
                                pcount+=1
                                break
                        if pcount==10:
                            break                
            if count==3:
                break 
                
posSentStr = ""
posSentStr = posSentStr.join(pos_sentences)
posTokenSentences = sent_tokenizer1.tokenize(posSentStr)
finResult(items, mc_pos, posTokenSentences)

negSentStr = ""
negSentStr = negSentStr.join(neg_sentences)
negTokenSentences = sent_tokenizer1.tokenize(negSentStr)
finResult(items, mc_neg, negTokenSentences)



