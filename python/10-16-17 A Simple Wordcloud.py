get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, get_single_color_func

# Config dataframe and plots
pd.set_option('max_columns', 999)
plt.rcParams["figure.figsize"] = [30,30]
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

text = '''In computer programming, pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. It is free software released under the three-clause BSD license.[2] The name is derived from the term "panel data", an econometrics term for multidimensional, structured data sets.'''


wordcloud = WordCloud(background_color="black", max_words=200,
                     width=2400, height=1600, margin = 20).generate(text)
plt.imshow(wordcloud, interpolation="nearest")

