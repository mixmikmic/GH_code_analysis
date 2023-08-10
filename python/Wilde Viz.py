import numpy as np
import csv
import random
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy.misc import imread
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

text = open('../Viz/_CLEAN_Wilde.txt').read()

image = imread('../Viz/green.png')

grn = ImageColorGenerator(image)

wilde_mask = imread('../Viz/wilde_sten2.png')

color='#DDA0DD'

wc = WordCloud(background_color=color, max_words=1000, mask=wilde_mask, stopwords=STOPWORDS)

wc.generate_from_text(text)

wc.recolor(color_func=grn, random_state=42)

plt.imshow(wc)
plt.axis('off')
plt.figure()

wc.to_file('../Viz/wilde_viz.png')



