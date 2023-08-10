from bs4 import BeautifulSoup  # For processing XMLfrom BeautifulSoup
import nltk
import re

doc = open('garden-party.xml').read()
soup = BeautifulSoup(doc, 'lxml')

segs = soup.findAll('seg')

text = ""
for seg in segs: 
    text += seg.text + " "

def cleanText(text): 
    text = text.replace('\n', ' ') # change newlines to spaces
    text = text.replace('\t', ' ') # change tabs to spaces
    text = re.sub('\s+', ' ', text).strip() # remove redundant whitespace
    return text

text = cleanText(text)

sents = nltk.sent_tokenize(text) # break the text up into sentences

len(sents) # how many sentences? 

ands = [sent for sent in sents if re.search(r'^And', sent) is not None]
ands # sentences that start with "And"

len(ands) # number of sentences that start with "And"

proportionOfAnds = (len(ands) / len(sents)) * 100 
proportionOfAnds # percentage of sentences that start with "And"

buts = [sent for sent in sents if re.search(r'^But', sent) is not None]
buts # sentences that start with "But"

len(buts)

(len(buts) / len(sents)) * 100



