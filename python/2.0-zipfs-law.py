import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

from src.data.loaders import load_and_clean_data
from src.data.preprocessor import Options
from src.definitions import TRAIN_PATH

ROWS = None # Load all reviews

options = (
    Options.EMAILS,
    Options.EMOTICONS,
    Options.PUNCTUATION,
    Options.REPEATING_VOWELS,
    Options.URLS
)

reviews, _ = load_and_clean_data(path=TRAIN_PATH, options=options, nrows=ROWS)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(reviews)

print('Total words: {}'.format(len(vectorizer.get_feature_names())))

reviews_matrix = vectorizer.transform(reviews)
word_count = np.sum(reviews_matrix, axis=0)
word_count = np.squeeze(np.asarray(word_count))

df = pd.DataFrame([word_count], columns=vectorizer.get_feature_names()).transpose()
df.rename({0: 'Count'}, axis='columns', inplace=True)
df.sort_values(['Count'], ascending=False, inplace=True)

df.head(20) # 20 most frequent words

n_words = 100
x = np.arange(n_words)
y = df.iloc[:n_words, 0]
most_frequent_word = y[0]
expected_zipf = [most_frequent_word / (i + 1) for i in range(n_words)]

plt.figure(figsize=(15, 8))
plt.bar(x, y, alpha=0.3, color='b')
plt.plot(x, expected_zipf, color='r', linestyle='--', linewidth=2, alpha=0.7)

plt.xlim([-1, n_words])
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('Top {} words in reviews'.format(n_words))
plt.legend(['Expected', 'Actual'])

