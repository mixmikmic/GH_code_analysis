import numpy as np
import csv
import random
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy.misc import imread
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

image = imread('../Viz/hard-sepia.png')

sep = ImageColorGenerator(image)

text = open('../Viz/_CLEAN_Twain.txt').read()

twain_mask = imread('../Viz/twain_sten2.png')

stopwords = STOPWORDS.copy()

stopwords.add('one')
stopwords.add('day')

wc = WordCloud(background_color='white', mask=twain_mask, max_words=300, stopwords=stopwords)

wc.generate_from_text(text)

wc.recolor(color_func=sep, random_state=42)

plt.imshow(wc)
plt.axis('off')
plt.figure()

wc.to_file('../Viz/twain_viz.png')



