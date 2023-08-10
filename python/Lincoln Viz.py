import numpy as np
import csv
import random
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

text = open('../Viz/lincoln.txt').read()

linc_mask = imread('../Viz/linc_sten.png')

def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(10, 50)

wc = WordCloud(background_color='white', mask=linc_mask, stopwords=STOPWORDS)

wc.generate(text)

wc.recolor(color_func=grey_color_func, random_state=3)

plt.imshow(wc)
plt.axis('off')
plt.figure()

wc.to_file('../Viz/linc_wc.png')



