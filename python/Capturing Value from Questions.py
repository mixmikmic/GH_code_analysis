import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords


df = pd.read_csv("qandas.csv")

print "The database df has",len(df), "number of rows."
df[0:10]

allwords = []
for items in df["Question"]:
    #print items
    temporar = []
    for words in items.split():
        words = words.strip(",[]\"'")
        temporar.append(words)
    allwords.append(temporar)

df1 = pd.DataFrame(np.zeros((169, 2)))
df1[0] = allwords
df1[1] = df["Images"]
print df1[0:10]

count = 0

hair_color = []

for items in df1[0]:
    items = " ".join(items)
    if "blond" in items:
        count+=1
        hair_color.append("blonde")
    elif "brown" in items:
        count+=1
        hair_color.append("brown")
        #print items
    elif "green hair" in items:
        count+=1
        hair_color.append("green")
        #print items
    elif "red hair" in items:
        count+=1
        hair_color.append("red")
        #print items
    elif "brunette" in items:
        count+=1
        hair_color.append("brown")
        #print itemsprint count
    else:
        hair_color.append("-")
print count

print hair_color[0:40]
print
print
print len(hair_color)



