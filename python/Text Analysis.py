import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
get_ipython().magic('matplotlib inline')

# Do a GET request to a web page
response = requests.get('http://www.paulgraham.com/hamming.html')
response

# Create a html parser
soup = BeautifulSoup(response.text, 'html.parser')

# Grab the text within the first table element from the web page
text = soup.table.get_text()
text

# Convert a collection of text documents to a matrix of token counts
vect = CountVectorizer(stop_words='english')
mat = vect.fit_transform([text])

# ngram
vect2 = CountVectorizer(stop_words='english', ngram_range=(2, 3))
mat2 = vect2.fit_transform([text])

# Get the counts to the first document
counts = mat.toarray()[0]
counts2 = mat2.toarray()[0]

# Find the index with the token 'work'
arr = np.asarray(vect.get_feature_names())
ind = np.where(arr == 'work')
counts[ind]

# Create feature array of (word, count)
features = list(zip(vect.get_feature_names(), counts))
features2 = list(zip(vect2.get_feature_names(), counts2))

import wordcloud

# Plot the word cloud
w = wordcloud.WordCloud()
img = w.generate_from_frequencies(features)
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(img)

img2 = w.generate_from_frequencies(features2)
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.show()



