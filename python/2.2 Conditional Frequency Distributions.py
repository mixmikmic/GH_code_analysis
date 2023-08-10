text = ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]

pairs = [('news', 'The'), ('news', 'Fulton'), ('news', 'County'), ...]
# each pair has the form (condition, event)

import nltk
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

genre_word = [(genre, word)
              for genre in ['news', 'romance']
              for word in brown.words(categories=genre)]
len(genre_word)

genre_word[:4]

genre_word[-4:]

cfd = nltk.ConditionalFreqDist(genre_word)
cfd

cfd.conditions()

print(cfd['news'])

print(cfd['romance'])

cfd['romance'].most_common(20)

cfd['romance']['could']

from nltk.corpus import inaugural
icfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target))

icfd.conditions()

years = []
for fileid in inaugural.fileids():
    years.append(fileid[:4])
#print(years)
last10 = years[-10:]
print(last10)

#icfd.tabulate()
icfd.tabulate(samples=last10)

from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch',
    'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
ucfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1'))

ucfd.tabulate(conditions=['English', 'German_Deutsch'],
             samples=range(10), cumulative=True)

cfd.conditions()

sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven',
        'and', 'the', 'earth', '.']
list(nltk.bigrams(sent))

def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

cfd['living']

cfd['creature']

cfd['thing']

generate_model(cfd, 'living')



