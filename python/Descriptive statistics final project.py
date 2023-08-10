import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

suits = ["Spades", "Hearts","Diamonds", "Clubs"]

unique_cards = {"Ace":1,"Two":2,"Three":3,"Four":4,"Five":5,"Six":6,"Seven":7,"Eight":8,"Nine":9,"Ten":10,"Jack":10,"Queen":10,
               "King":10}

# Creates the Deck
Deck = {}

for i in range(4):
    for key,val in unique_cards.items():
        Deck[key + " of " + suits[i] + " "] = val

df = pd.DataFrame({"Cards": Deck.keys(), "Values":Deck.values()})

df

df["Values"].describe()


df["Values"].hist(bins=10)

samples_sum = []

for i in range(30):
    item = df.sample(n=3,replace = False)["Values"].sum()
    samples_sum.append(item)

print samples_sum

samples_sum = pd.Series(samples_sum)
print "median " + str(samples_sum.median())
samples_sum.describe()

samples_sum.hist(bins=10)

#For the top 5% or bottom 5% the corresponding Z score would be

z = 1.645

# Calculating SE from the original dataframe's std with a sample size of 3.
    
print samples_sum.mean() + z*samples_sum.std()
print samples_sum.mean() - z*samples_sum.std()


df["Values"].describe()

(20-samples_sum.mean())/samples_sum.std()

# From Z chart we get 

p = 1- 0.47934 # for Z = -0.35906

# 0.47934 is the probability that the drawn sample will be less than or equal to 20, so we do 1 - 0.47934 to get the probability 
# that the drawn sample will be larger than p.

p



