with open('../data/corpora/enlang1.txt') as fin:
    print(fin.readlines()[0][:200])

V = 0 #size of vocabulary
histogram = {} #unigram and bigram frequencies

with open('../data/corpora/enlang1.txt') as fin:
    for doc in fin.readlines():
        for i in range(len(doc)-2):
            bigram = doc[i:i+2]
            unigram = doc[i]
            histogram[bigram] = histogram.get(bigram, 0) + 1
            histogram[unigram] = histogram.get(unigram, 0) + 1
    V = len([unigram for unigram in histogram.keys() if len(unigram) == 1])

print(V)
print(histogram['en'])
print(histogram['e'])

import numpy as np

#Compute the probability of a bigram using the Laplace smoothing
def getProbability(bigram):
    return 1.0*(histogram.get(bigram, 0) + 1) /                 (histogram.get(bigram[0], 0) + V)

# Get the perplexity of text.
def getPerplexity(text):
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    h = -sum(map(lambda x: np.log2(getProbability(x)), bigrams))
    return np.power(2, h/len(bigrams))

PATH = '../data/corpora/'

languages = {'en': {'file': 'enlang2.txt'},
             'cs': {'file': 'cslang.txt'},
             'es': {'file': 'eslang.txt'},
             'fr': {'file': 'frlang.txt'},
             'it': {'file': 'itlang.txt'},
             'ru': {'file': 'rulang.txt'},
             'mixed': {'file': 'mixedlang.txt'}
            }

for lang, v in languages.items():
    with open(PATH + v['file']) as fin:
        perplexities = []
        for doc in fin.readlines():
            doc = doc.strip()
            perplexities.append(getPerplexity(doc))
        languages[lang]['perplexities'] = perplexities

get_ipython().magic('matplotlib notebook')

import matplotlib.pyplot as plt

x1 = languages['en']['perplexities']
x2 = languages['mixed']['perplexities']

mu1 = np.mean(x1)
sigma1 = np.std(x1)
print("Mean and standard deviation of the first dataset: mean={}, std={}.".format(mu1, sigma1))

mu2 = np.mean(x2)
sigma2 = np.std(x2)
print("Mean and standard deviation of the second dataset: mean={}, std={}.".format(mu2, sigma2))

# histograms of perplexities
plt.hist(x1, bins='auto', normed=1, facecolor='blue', alpha=0.8)
plt.hist(x2, bins=1000, normed=1, facecolor='red', alpha=0.8)

plt.xlabel('Perplexity')
plt.ylabel('Density')
plt.axis([8, 20, 0, 0.55])
plt.grid(True)

plt.show()

def detectLang(text, threshold=14):
    text = text.lower() #The training corpus was also in the lowercase form.
    if len(text) <= 1:
        return False
    else:
        return True if getPerplexity(text) <= threshold else False

print(detectLang('this is an example of the english language'))
print(detectLang('another text written in the target language which should pass'))
print(detectLang('toto je ukázkový český text'))
print(detectLang('následuje alternativní posloupnost znaků'))



