

import nltk, re, pprint

nltk.download()

nltk.data.path.append("C:/nltk_data/")

from nltk.book import *

text1

len(text4)

sorted(set(text4))

text4.count("democracy")



myFreqDist = FreqDist(text4)

myFreqDist.most_common(100)

from nltk.corpus import stopwords # Import the stop word list
print (stopwords.words("english"))

reducedWords = [w for w in text4 if not w in stopwords.words("english")]
print (reducedWords)

myFreqDistReduced = FreqDist(reducedWords)

myFreqDistReduced.most_common(10);

long_words = [w for w in myFreqDistReduced if len(w) > 15]

print(long_words);

myWordList = [w for w in reducedWords if re.search("[a-zA-Z]", w)]

myFreqDistReduced = FreqDist(myWordList)

myFreqDistReduced.most_common(10);

text4.findall(r"<a> (<.*>) <world>")

text4.findall(r"<.*> <.*> <world>");

text4.findall(r"<world> <.*> <.*> ");

text4.concordance("world");

text4.similar("world")

get_ipython().run_line_magic('matplotlib', 'inline')

text4.dispersion_plot(["world", "democracy", "freedom", "duties", "America"]);

text1.dispersion_plot(["whale", "coast", "sea", "shore", "angry"])

text4.dispersion_plot(["terror", "fear", "Isis", "Iraq", "depression", "Britain"])

text4.dispersion_plot(["freedom", "responsibility", "justice"])

def graph(myFile):
  f = open(myFile, "r")
  inputfile = f.read()
  tokens = nltk.tokenize.word_tokenize(inputfile)
  fd = nltk.FreqDist(tokens)
  fd.plot(30,cumulative=False)

myFile = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')

graph(myFile)



