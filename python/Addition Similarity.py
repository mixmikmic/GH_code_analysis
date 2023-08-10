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
def similarity(additionA, additionB):
    diff = np.square(wtb[additionA] - wtb[additionB])
    return diff.mean()

res = []
# Loop through each addition pair
for additionA in wtb.columns:
    for additionB in wtb.columns:
        # Skip if additionA and combo B are the same. 
        # To prevent duplicates, skip if A is after B alphabetically
        if additionA != additionB and additionA < additionB:
            res.append([additionA, additionB, similarity(additionA, additionB)])
df = pd.DataFrame(res, columns=["additionA", "additionB", "similarity"])

df.sort_values("similarity").head(10)

df.sort_values("similarity", ascending=False).head(10)

def comboSimilarity(additionA, additionB):
    # additionA needs to be before additionB alphabetically
    if additionA > additionB:
        addition_temp = additionA
        additionA = additionB
        additionB = addition_temp
    return df.loc[df['additionA'] == additionA].loc[df['additionB'] == additionB]
comboSimilarity('plum', 'vanilla')

df.describe()

