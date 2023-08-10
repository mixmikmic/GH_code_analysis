import math

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def logistic(x):
    return 1 / (1 + math.exp(-x))

# Get data
data = {
    'hours': [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5],
    'pass' : [0  , 0   , 0, 0   , 0  , 0   , 1, 0   , 1  , 0   , 1, 0   , 1  , 0, 1   , 1  , 1   , 0, 1]}
df = pd.DataFrame(data)
df.head()

df.plot(x='hours', y='pass', kind='scatter', color='red')

# P(Y | X1) = 1 / ( 1 + e^(-(mx + c)) )

# assume after training, coefficients are as follows:
m = 1.5
c = -4

m = 1.5
c = -4
hrs = 3
probability = m*hrs + c

1 / ( 1 + math.exp(-probability))

predicted = []
for h in data['hours']:
    probability = m*h + c
#     probability = logistic(probability)
    predicted.append(probability)

df['predicted'] = predicted

ax = df.plot(x='hours', y='pass', kind='scatter', color='red')
df.plot(x='hours', y='predicted', ax=ax, color='blue')
plt.axhline(y=0.5, color='grey')




