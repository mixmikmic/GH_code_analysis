# Imports
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import json
sns.set()

# Get datasets into DataFrames
df = pd.read_csv('data/tweets1.csv')
df2 = pd.read_csv('data/tweets2.csv')

# Get English language tweets, split into words and remove stopwords
text = df[df.lang == 'en'].text.str.cat(sep = ' ').lower()
stopwords = set(STOPWORDS)
stopwords.update(['https', 'RT', 'co'])

# Generate wordcloud and save to file
wordcloud = WordCloud(width=1000, height=500, max_font_size=90, collocations=False, stopwords=stopwords).generate(text)
wordcloud.to_file("wc1.png")

# Display the generated image:
# the matplotlib way:
#import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
#plt.figure(figsize=(8, 6))
plt.axis("off")

# Get English language tweets, split into words and remove stopwords
text = df2[df2.lang == 'en'].text.str.cat(sep = ' ').lower()
stopwords = set(STOPWORDS)
stopwords.update(['https', 'RT', 'co'])

# Generate wordcloud and save to file
wordcloud = WordCloud(width=1000, height=500, max_font_size=90, collocations=False, stopwords=stopwords).generate(text)
wordcloud.to_file("wc2.png")

# Display the generated image:
# the matplotlib way:
#import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
#plt.figure(figsize=(8, 6))
plt.axis("off")

