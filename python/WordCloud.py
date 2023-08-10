from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

"""
Data fetched from
https://www.kaggle.com/crowdflower/twitter-airline-sentiment/data
"""
df = pd.read_csv("Tweets.csv")
df.shape

df.head()

text = ""
for ind, row in df.iterrows():
    text += row["text"] + " "
text = text.strip()

wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=100, max_words=40).generate(text)
wordcloud.recolor(random_state=ind*312)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

