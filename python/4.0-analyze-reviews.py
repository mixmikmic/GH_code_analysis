import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

get_ipython().magic('matplotlib inline')

data = pd.read_csv("../data/processed/train.csv", header=None, na_filter=False)

pos = data[data[0] == 1]
neg = data[data[0] == 0]

pos_percent = 100 * pos.shape[0] / data.shape[0]
neg_percent = 100 * neg.shape[0] / data.shape[0]

plt.figure(figsize=(15,10))
plt.bar([0, 1, 2], [len(data), len(pos), len(neg)], color=['g', 'b', 'r'])
plt.xticks([0, 1, 2], ['Total', 'Positive ({:.2f}%)'.format(pos_percent), 'Negative ({:.2f}%)'.format(neg_percent)])
plt.show()

get_ipython().run_cell_magic('capture', '', 'from keras.preprocessing.text import Tokenizer\n\npos_reviews = (pos[1] + " " + pos[2]).values\npos_tokenizer = Tokenizer(num_words=100)\n\npos_tokenizer.fit_on_texts(pos_reviews)')

from keras.preprocessing.text import Tokenizer

neg_reviews = (neg[1] + " " + neg[2]).values
neg_tokenizer = Tokenizer(num_words=100)

neg_tokenizer.fit_on_texts(neg_reviews)

from nltk.corpus import stopwords

stopset = set(stopwords.words('english'))

def sorted_words(tokenizer, opposite_tokenizer=None):
    if opposite_tokenizer:
         tokens = [(word, frequency) for word, frequency in tokenizer.word_counts.items() 
                   if word not in stopset and word not in opposite_tokenizer.word_counts]
    else:
        tokens = [(word, frequency) for word, frequency in tokenizer.word_counts.items() 
                  if word not in stopset]
        
    sorted_words = sorted(tokens, key=lambda item: item[1], reverse=True)
    return list(sorted_words)

from wordcloud import WordCloud

pos_dict = dict((x, y) for x, y in sorted_words(pos_tokenizer))
pos_word_cloud = WordCloud(background_color="white", max_words=10)
pos_word_cloud.fit_words(pos_dict)

plt.figure(figsize=(15, 10))
plt.imshow(pos_word_cloud)
plt.axis("off")
plt.show()

from wordcloud import WordCloud

neg_dict = dict((x, y) for x, y in sorted_words(neg_tokenizer))
neg_word_cloud = WordCloud(background_color="white", max_words=10)
neg_word_cloud.fit_words(neg_dict)

plt.figure(figsize=(15, 10))
plt.imshow(neg_word_cloud)
plt.axis("off")
plt.show()

from wordcloud import WordCloud

pos_only_dict = dict((x, y) for x, y in sorted_words(pos_tokenizer, neg_tokenizer))
pos_only_word_cloud = WordCloud(background_color="white", max_words=10)
pos_only_word_cloud.fit_words(pos_only_dict)

plt.figure(figsize=(15, 10))
plt.imshow(pos_only_word_cloud)
plt.axis("off")
plt.show()

from wordcloud import WordCloud

neg_only_dict = dict((x, y) for x, y in sorted_words(neg_tokenizer, pos_tokenizer))
neg_only_word_cloud = WordCloud(background_color="white", max_words=10)
neg_only_word_cloud.fit_words(neg_only_dict)

plt.figure(figsize=(15, 10))
plt.imshow(neg_only_word_cloud)
plt.axis("off")
plt.show()

get_ipython().run_cell_magic('capture', '', 'from keras.preprocessing.text import text_to_word_sequence\n\n\nnumber_of_words = [len(text_to_word_sequence(row[1] + " " + row[2]))\n                   for row in data.itertuples(index=False, name=None)]')

plt.boxplot(number_of_words)

ndf = pd.DataFrame({'words': number_of_words})
ndf.describe()

plt.figure(figsize=(15, 10))
plt.hist(number_of_words, orientation='horizontal', rwidth=0.95)

bow_data = pd.read_csv("../reports/f1_score_bow_diff_options.csv")

X_bow = bow_data["train_size"].values
y_bow = bow_data.drop(['train_size'], axis=1).values[0]
colors = ['r', 'b', 'g', 'm', 'c', '#feff60', '#007485', '#7663b0', '#f47be9']
labels = ["Emails + Urls", "Stopwords", "Emoticons", "Lemmatizer", "Punctuation", 
          "Repeating vowels", "Stemmer", "Spelling","Negative constructs"]

plt.figure(figsize=(15, 10))
plt.bar(range(len(y_bow)), y_bow, color=colors)
plt.xticks(range(len(y_bow)), labels)
plt.ylim(min(y_bow) - 0.001, max(y_bow) + 0.0005)

plt.ylabel("Accuracy")
plt.show()

