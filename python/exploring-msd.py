get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
from sklearn import preprocessing
from keras import regularizers
from keras.utils import np_utils, generic_utils

import pandas as pd
from pandas import compat
compat.PY3 = True

df = pd.read_csv("/mnt/c/Users/Aumit/Documents/GitHub/million-song-analysis/output.csv")

df.keys()

years = df['year'].tolist()

hotttness = df['hotttness'].tolist()

plt.plot(years, hotttness, '.')
plt.title("Hotttness over the Decades")
plt.xlabel("Decades")
plt.ylabel("Hotttness")
plt.xlim([1940,2011])
#plt.plot(years, m*years + b, '-')

familiarity = df['familiarity'].tolist()

plt.plot(years, familiarity, '.')
plt.title("Familiarity over the Decades")
plt.xlabel("Decades")
plt.ylabel("Familiarity")
plt.xlim([1940,2011])

tempo = df['tempo'].tolist()

plt.plot(years, tempo, '.')
plt.title("Tempo over the Decades")
plt.xlabel("Decades")
plt.ylabel("Tempo")
plt.xlim([1940,2011])

loudness = df['loudness'].tolist()

plt.plot(years, loudness, '.')
plt.title("Loudness over the Decades")
plt.xlabel("Decades")
plt.ylabel("Loudness")
plt.xlim([1940,2011])

artist_names = df['artist_name'].tolist()

artist_name_lenghts = [len(str(name)) for name in artist_names]

len(artist_name_lenghts)

plt.plot(years, artist_name_lenghts, '.')
plt.title("Length of Artist's Names over the Decades")
plt.xlabel("Decades")
plt.ylabel("Lenght of Artist's Name")
plt.xlim([1940,2011])
plt.ylim([0,50])

