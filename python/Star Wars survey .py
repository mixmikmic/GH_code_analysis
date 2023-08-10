import pandas as pd
star_wars  = pd.read_csv("star_wars.csv",encoding="ISO-8859-1")

star_wars.head()

star_wars.columns

star_wars = star_wars[pd.notnull(star_wars["RespondentID"])]

star_wars["RespondentID"].isnull().sum()

star_wars['Have you seen any of the 6 films in the Star Wars franchise?'].value_counts()

yes_no = {"Yes":True,"No":False}

for col in ['Have you seen any of the 6 films in the Star Wars franchise?',
        'Do you consider yourself to be a fan of the Star Wars film franchise?']:
    star_wars[col]= star_wars[col].map(yes_no)
    

print star_wars['Do you consider yourself to be a fan of the Star Wars film franchise?'].value_counts()
print star_wars['Have you seen any of the 6 films in the Star Wars franchise?'].value_counts()

import numpy as np

movie_mapping = {
    "Star Wars: Episode I  The Phantom Menace": True,
    np.nan: False,
    "Star Wars: Episode II  Attack of the Clones": True,
    "Star Wars: Episode III  Revenge of the Sith": True,
    "Star Wars: Episode IV  A New Hope": True,
    "Star Wars: Episode V The Empire Strikes Back": True,
    "Star Wars: Episode VI Return of the Jedi": True
}

for col in star_wars.columns[3:9]:
    star_wars[col]=star_wars[col].map(movie_mapping)

for col in star_wars.columns[3:9]:
    print(star_wars[col].value_counts())

star_wars = star_wars.rename(columns = {
       'Which of the following Star Wars films have you seen? Please select all that apply.':"seen_1",
       'Unnamed: 4':"seen_2", 
       'Unnamed: 5':"seen_3", 
       'Unnamed: 6':"seen_4", 
       'Unnamed: 7':"seen_5",
       'Unnamed: 8':"seen_6"})


star_wars.head()

star_wars[star_wars.columns[9:15]] = star_wars[star_wars.columns[9:15]].astype(float)

star_wars = star_wars.rename(columns={
        "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "ranking_1",
        "Unnamed: 10": "ranking_2",
        "Unnamed: 11": "ranking_3",
        "Unnamed: 12": "ranking_4",
        "Unnamed: 13": "ranking_5",
        "Unnamed: 14": "ranking_6"
        })

star_wars.head()

star_wars[star_wars.columns[9:15]].mean()

get_ipython().magic('matplotlib inline')
import seaborn as sns
import matplotlib.pyplot as plt

index = ["Star Wars: Episode I  The Phantom Menace",
    "Star Wars: Episode II  Attack of the Clones",
    "Star Wars: Episode III  Revenge of the Sith",
    "Star Wars: Episode IV  A New Hope",
    "Star Wars: Episode V The Empire Strikes Back",
    "Star Wars: Episode VI Return of the Jedi"]

plt.bar(range(6), star_wars[star_wars.columns[9:15]].mean())
plt.xticks(range(6),index,rotation = "vertical")
plt.xlabel("Star Wars movies preference chart(1:most favorite, 5: least favorite)")
plt.ylabel("Rating from surveymonkey users")

star_wars[star_wars.columns[9:15]].sum()

plt.bar(range(6), star_wars[star_wars.columns[9:15]].sum())
plt.xticks(range(6),index,rotation = "vertical")
plt.xlabel("Star Wars movies viewer chart")
plt.ylabel("Number of viewers who have seen that movie")

males = star_wars[star_wars["Gender"] == "Male"]
females = star_wars[star_wars["Gender"] == "Female"]

males[males.columns[9:15]].mean()

plt.bar(range(6), males[males.columns[9:15]].mean())
plt.xticks(range(6),index,rotation = "vertical")
plt.xlabel("Star Wars movies preference chart for males(1:most favorite, 5: least favorite)")
plt.ylabel("Rating from surveymonkey users")

females[females.columns[9:15]].mean()

plt.bar(range(6), females[females.columns[9:15]].mean())
plt.xticks(range(6),index,rotation = "vertical")
plt.xlabel("Star Wars movies preference chart for female viewers(1:most favorite, 5: least favorite)")
plt.ylabel("Rating from surveymonkey users")

N = 6
menMeans = males[males.columns[9:15]].mean()


ind = np.arange(N)  # the x locations for the groups
width = 0.40       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r')

womenMeans = females[females.columns[9:15]].mean()
rects2 = ax.bar(ind + width, womenMeans, width, color='y')

#add some text for labels, title and axes ticks
ax.set_ylabel('Star wars movie rating by gender(1:most favorite,5:least favorite)')
ax.set_xticks(ind + width)
ax.set_xticklabels(index, rotation = "vertical")

ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))






plt.show()

N = 6
menMeans = males[males.columns[9:15]].sum()


ind = np.arange(N)  # the x locations for the groups
width = 0.40       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r')

womenMeans = females[females.columns[9:15]].sum()
rects2 = ax.bar(ind + width, womenMeans, width, color='y')

#add some text for labels, title and axes ticks
ax.set_ylabel('Star wars viewer counts')
ax.set_xticks(ind + width)
ax.set_xticklabels(index, rotation = "vertical")

ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))





plt.show()



