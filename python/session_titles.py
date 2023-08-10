get_ipython().system('pip install wordcloud')
get_ipython().magic('matplotlib inline')

from os import path
from wordcloud import WordCloud


# Read the whole text.
text = open('session_titles.tsv').read().lower()

# Generate a word cloud image
wordcloud = WordCloud(width=3000, height=2000, prefer_horizontal=1, stopwords=None).generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.figure(figsize=(30,15))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("cloud2.png")



