from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Mecab
import re

mecab = Mecab()

def text_cleaning(text):
    text = re.sub('[^가-힝0-9a-zA-Z]', ' ', text)
    return text

def top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        keywords = [" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])]
    return keywords

with open('source/2016huna1.txt') as text:
    corpus = text.read()

data = text_cleaning(corpus)
data = mecab.nouns(data)

vectorizer = TfidfVectorizer(lowercase=False)
x_list = vectorizer.fit_transform(data)

keywords = []
lda = LatentDirichletAllocation(n_topics=3, learning_method='batch').fit(x_list)
feature_names = vectorizer.get_feature_names()
keywords.append(top_words(lda, feature_names, 5))

keywords

