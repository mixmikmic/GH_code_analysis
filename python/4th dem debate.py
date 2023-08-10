import requests
from lxml import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pprint import pprint
import re

get_ipython().magic('matplotlib inline')

url = "https://www.washingtonpost.com/news/the-fix/wp/2016/01/17/the-4th-democratic-debate-transcript-annotated-who-said-what-and-what-it-meant/"
response = requests.get(url)

doc = html.fromstring(response.text)

para_list = doc.xpath("//article/p/text()")

para_list = para_list[2:]

pprint(para_list[:2], compact=True)
print(para_list[-2:])

dataset = pd.DataFrame(para_list, columns=["raw"])
dataset

def get_name(x):
    r = re.findall(r"^([A-Z']*):", x)
    if r:
        return r[0]
    else:
        return np.NaN

dataset["speaker"] = dataset.raw.apply(get_name).fillna(method='ffill')
dataset

dataset.speaker.value_counts()

get_speach = lambda x: re.sub("^[A-Z']*:\s", "", x)
dataset["speach"] = dataset.raw.apply(get_speach)
dataset

applause_ds = dataset[dataset.speach == "(APPLAUSE)"]
len(applause_ds)

applause_ds.speaker.value_counts()

applause_counts = applause_ds.speaker.value_counts().sort_values()

bottom = [index for index, item in enumerate(applause_counts.index)]
plt.barh(bottom, width=applause_counts, color="orange", linewidth=0)

y_labels = ["%s %.1f%%" % (item, 100.0*applause_counts[item]/len(applause_ds)) for index,item in enumerate(applause_counts.index)]
plt.yticks(np.array(bottom)+0.4, y_labels)

applause_counts

word_count = lambda x: len(re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",x))

dataset["word_count"] = dataset.speach.apply(word_count)
dataset

words_ds = dataset[dataset.speaker.isin(["CLINTON","SANDERS","O'MALLEY"])]

words_counts = words_ds.pivot_table(values="word_count", index="speaker", columns=None, aggfunc='mean',).sort_values()

bottom = [index for index, item in enumerate(words_counts.index)]
plt.barh(bottom, width=words_counts, color="orange", linewidth=0)

y_labels = ["%s %.1f words/paragraph" % (item, words_counts[item]) for index,item in enumerate(words_counts.index)]
plt.yticks(np.array(bottom)+0.4, y_labels)

words_counts

words_counts = words_ds.pivot_table(values="word_count", index="speaker", columns=None, aggfunc='sum',).sort_values()

bottom = [index for index, item in enumerate(words_counts.index)]
plt.barh(bottom, width=words_counts, color="orange", linewidth=0)

y_labels = ["%s %d (%.1f%%)" % (item, words_counts[item], 100.0*words_counts[item]/np.sum(words_counts)) for index,item in enumerate(words_counts.index)]
plt.yticks(np.array(bottom)+0.4, y_labels)

words_counts

speaker_dict = {value:index for index,value in enumerate(words_ds.speaker.unique())}
speaker_dict

words_ds["speaker_no"] = words_ds.speaker.map(speaker_dict)
words_ds

cv = CountVectorizer()
count_matrix = cv.fit_transform(words_ds.speach)
count_matrix = count_matrix.toarray()

word_count = pd.DataFrame(cv.get_feature_names(), columns=["word"])
word_count["count"] = count_matrix.sum(axis=0)
word_count = word_count.sort_values(by="count", ascending=False).reset_index(drop=True)
word_count[:]

cl = MultinomialNB()
cl.fit(count_matrix, words_ds.speaker=="SANDERS")

df_vocab = pd.DataFrame(list(cv.vocabulary_.keys()), columns=["Vocab"])
df_vocab["Vocab_index"] = cv.vocabulary_.values()
df_vocab = df_vocab.sort_values("Vocab_index").reset_index(drop=True)
df_vocab["proba"] = cl.feature_log_prob_[0]
df_vocab["anti_proba"] = cl.feature_log_prob_[1]
df_vocab["difference"] = cl.feature_log_prob_[0] - cl.feature_log_prob_[1]
df_vocab.sort_values("difference", ascending=True)

cl = MultinomialNB()
cl.fit(count_matrix, words_ds.speaker=="CLINTON")

df_vocab = pd.DataFrame(list(cv.vocabulary_.keys()), columns=["Vocab"])
df_vocab["Vocab_index"] = cv.vocabulary_.values()
df_vocab = df_vocab.sort_values("Vocab_index").reset_index(drop=True)
df_vocab["proba"] = cl.feature_log_prob_[0]
df_vocab["anti_proba"] = cl.feature_log_prob_[1]
df_vocab["difference"] = cl.feature_log_prob_[0] - cl.feature_log_prob_[1]
df_vocab.sort_values("difference", ascending=True)

cl = MultinomialNB()
cl.fit(count_matrix, words_ds.speaker=="O'MALLY")

df_vocab = pd.DataFrame(list(cv.vocabulary_.keys()), columns=["Vocab"])
df_vocab["Vocab_index"] = cv.vocabulary_.values()
df_vocab = df_vocab.sort_values("Vocab_index").reset_index(drop=True)
df_vocab["proba"] = cl.feature_log_prob_[0]
df_vocab["anti_proba"] = cl.feature_log_prob_[1]
df_vocab["difference"] = cl.feature_log_prob_[0] - cl.feature_log_prob_[1]
df_vocab.sort_values("difference", ascending=True)



