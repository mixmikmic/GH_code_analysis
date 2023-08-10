import os, sys
import pandas as pd
import nltk
from IPython.display import display
import json
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Define the absolute path of directory where the data has been stored 
data_path = "/home/gkastro/title-prediction-tensorflow/content-data/"
os.chdir(data_path)

# You only need to run this block once, then you can start without executing this one since the DataFrame 
# will already exist as a pickle file.

title_df = pd.DataFrame(columns=["Year", "Month", "Title"])
for year in range(2008, 2018):
    for month in range(1, 13):
        path = str(year)+"/"+str(month)+"/"
        for filename in os.listdir(path):
            file = open(path+filename, "r")
            content = json.load(file)
            title = content["title"]["title"]
            row = pd.DataFrame([[year, month, title]], columns=["Year", "Month", "Title"])
            title_df = title_df.append(row)

            title_df["Year"] = title_df["Year"].apply(int)
            
title_df["Month"] = title_df["Month"].apply(int)
title_df.to_pickle("title-dataframe")
title_df = pd.read_pickle("title-dataframe")
t_df = title_df.copy()
tmp = pd.Series(t_df["Title"].apply(len), name="Title-length")
t_df.insert(len(t_df.columns), tmp.name, tmp.values)
t_df.to_pickle("title-dataframe-w-length")

t_df = pd.read_pickle("title-dataframe-w-length")
t_df.head()

word = "brexit"
groups = t_df.groupby(["Year", "Month"])["Title"]
data = []
dates = []
for group in groups:
    count=0
    for title in group[1]:
        if word.lower() in title.lower().split(" "):
            count +=1
#     display(str(group[0][0])+"--"+str(group[0][1])+"--"+str(count))
    data.append(count)
    dates.append(str(group[0][0])+"/"+str(group[0][1]))
fig = plt.figure(figsize=(20,12))
plt.plot(pd.to_datetime(dates), data)
plt.title("Occurences of word: "+word+" , in FT article titles")
plt.grid()
plt.show()

data = t_df.groupby(["Year","Month"])["Title-length"].median()
fig = plt.figure(figsize=(20,12))
plt.plot(pd.to_datetime(dates), data)
plt.title("Median length (# of characters) in FT article titles throughout the years")
plt.grid()
plt.show()

