import nltk

nltk.corpus.gutenberg.fileids()
# from nltk.corpus import gutenberg
# gutenberg.fileids()

emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)

raw_txt = nltk.corpus.gutenberg.raw('austen-emma.txt')
words_txt = nltk.corpus.gutenberg.words('austen-emma.txt')
sents_txt = nltk.corpus.gutenberg.sents('austen-emma.txt')

print('-----'*5)
print('Raw Text :')
print(raw_txt[:10])
print('-----'*5)
print('Word Text :')
print(words_txt[:10])
print('-----'*5)
print('Sent Text :')
print(sents_txt[:3])
print('-----'*5)

from nltk.corpus import webtext
from nltk.corpus import nps_chat
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
chatroom[123]

from nltk.corpus import brown
brown.categories()

brown.words(categories='news')

from nltk.corpus import reuters
reuters.fileids()[:5]

reuters.categories()[:10]

reuters.words(categories='coffee')

from nltk.corpus import inaugural
inaugural.fileids()[-5:]

words = nltk.corpus.words.words() 
words[:20]

from nltk.corpus import stopwords
stopwords.words('english')[:10]













