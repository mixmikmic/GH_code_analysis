test_sentence = "Arthur, the fun westie, rolled on the grass while the squirrel sneeked past."

# 1
test_lower = test_sentence.lower()
print(test_lower)

import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('brown')
#nltk.download('averaged_perceptron_tagger')

stop_words = set(nltk.corpus.stopwords.words('english'))
word_tokens = nltk.tokenize.word_tokenize(test_lower)
print(test_lower.split())
print(word_tokens)
print([word for word in word_tokens if word not in stop_words])
filtered_sentence = [word for word in word_tokens if word not in stop_words]

# Tagger
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

backoff = nltk.RegexpTagger([
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
        (r'(The|the|A|a|An|an)$', 'AT'),   # articles
        (r'.*able$', 'JJ'),                # adjectives
        (r'.*ness$', 'NN'),                # nouns formed from adjectives
        (r'.*ly$', 'RB'),                  # adverbs
        (r'.*s$', 'NNS'),                  # plural nouns
        (r'.*ing$', 'VBG'),                # gerunds
        (r'.*ed$', 'VBD'),                 # past tense verbs
        (r'.*', 'NN')                      # nouns (default)
        ])
train_sents = brown_tagged_sents[:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=backoff)



#t3 = nltk.tag.brill.BrillTaggerTrainer()

#t3 = nltk.tag.brill.BrillTagger(t2,)
#t2.tag(filtered_sentence)


# Part of speech tagger, best one
nltk.pos_tag(filtered_sentence)

# going to use TextBlob instead of nltk
from textblob import TextBlob
import textblob

blob = TextBlob(test_sentence, pos_tagger=textblob.taggers.NLTKTagger())

print(blob.tags)
d = dict(blob.tags)
print(blob.noun_phrases)
print(blob.words)
print([(' '.join(phrase), ' '.join(map(lambda x: d[x],phrase))) for phrase in blob.ngrams(n=3)])

for sentence in blob.sentences:
    print(sentence.sentiment.polarity)
    
# get rid of all stop words, don't grab NNP for now, grab NN
protected_words = ' '.join(blob.noun_phrases).split()
sentence_filter2 = [word for word, pos in blob.tags if pos in ['NN']] # ,'VBN','VBD'
print(sentence_filter2)
filter_sentence_final = []
for phrase in blob.noun_phrases:
    if len(set([word for word in sentence_filter2 if word in phrase])) == len(set(phrase.split())):
        filter_sentence_final.append(phrase)
filter_sentence_final.extend(list(set(sentence_filter2) - set(protected_words)))
print(filter_sentence_final)

from google_images_download import google_images_download
response = google_images_download.googleimagesdownload() 
arguments = {"keywords":"Polar bears,baloons,Beaches","limit":20,"print_urls":True}
response.download(arguments)



