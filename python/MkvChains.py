import pensieve as pens
import textacy
from collections import defaultdict
from random import random

def make_markov_chain(docs):
    my_dict = defaultdict(list)
    inverse_dict = defaultdict(list)
    for doc in docs:
        print("Reading ",doc)
        d = pens.Doc(doc)
        for p in d.paragraphs:
            for sent in p.doc.sents:
                #print(sent.text)
                bow = textacy.extract.words(sent)
                for i_word, word in enumerate(bow):
                    if i_word < 3:
                        continue
                    key = sent[i_word-2].text+' '+sent[i_word-1].text
                    value = sent[i_word].text
                    my_dict[key].append(value)
                    inverse_dict[value].append(key)
    return my_dict, inverse_dict

def sample_from_chain(mv_dict, key):
    options = len(mv_dict[key])
    x = 999
    while x > options-1:
        x = int(10*(random()/options)-1)
    #rint(x)
    #print(x,key, options)
    return(mv_dict[key][x])

def make_chain(mkv_chain, key):
    counter = 0
    chain = key
    while key in mkv_chain:
        #if counter > 5:
        #    return chain
        chain+=' '+sample_from_chain(mkv_chain,key)
        key = chain.split()[-2]+' '+chain.split()[-1]
        counter +=1
    return chain
     

all_books = ['../../clusterpot/book1.txt',
            '../../clusterpot/book2.txt',
            '../../clusterpot/book3.txt',
            '../../clusterpot/book4.txt',
            '../../clusterpot/book5.txt',
            '../../clusterpot/book6.txt',
            '../../clusterpot/book7.txt']


mkv_chain, inv_chain = make_markov_chain(all_books)

#print(mkv_chain)
for i in range(20):
    print('\n',make_chain(mkv_chain,'He said'))

















