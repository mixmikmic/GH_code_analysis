# Style Similarity

# Import libraries
import numpy as np
import pandas as pd
# Import the data
import WTBLoad
wtb = WTBLoad.load()

import math
# Square the difference of each row, and then return the mean of the column. 
# This is the average difference between the two.
# It will be higher if they are different, and lower if they are similar
def similarity(styleA, styleB):
    diff = np.square(wtb[styleA] - wtb[styleB])
    return diff.mean()

res = []
# Loop through each addition pair
wtb = wtb.T
for styleA in wtb.columns:
    for styleB in wtb.columns:
        # Skip if styleA and combo B are the same. 
        # To prevent duplicates, skip if A is after B alphabetically
        if styleA != styleB and styleA < styleB:
            res.append([styleA, styleB, similarity(styleA, styleB)])
df = pd.DataFrame(res, columns=["styleA", "styleB", "similarity"])

df.sort_values("similarity").head(10)

df.sort_values("similarity", ascending=False).head(10)

def comboSimilarity(styleA, styleB):
    # styleA needs to be before styleB alphabetically
    if styleA > styleB:
        addition_temp = styleA
        styleA = styleB
        styleB = addition_temp
    return df.loc[df['styleA'] == styleA].loc[df['styleB'] == styleB]
comboSimilarity('Blonde Ale', 'German Pils')

df.describe()

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt

n, bins, patches = plt.hist(df['similarity'], bins=50)

similarity = float(comboSimilarity('Blonde Ale', 'German Pils')['similarity'])

# Find the histogram bin that holds the similarity between the two
target = np.argmax(bins>similarity)
patches[target].set_fc('r')
plt.show()

