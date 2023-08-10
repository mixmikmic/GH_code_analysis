import pandas as pd
star_wars = pd.read_csv("star_wars.csv", encoding="ISO-8859-1")

star_wars.columns

# star_wars['RespondentID'].dropna(axis=0, inplace=True)
# star_wars.shape

star_wars = star_wars[pd.notnull(star_wars["RespondentID"])]
star_wars.shape

import numpy as np
yes_no={'Yes':True,'No':True}
for col in star_wars.columns[1:3]:
    star_wars[col]=star_wars[col].map(yes_no)
    


star_wars.head()

type(star_wars)

star_wars.info()



star_wars[star_wars.columns[3:9]]

#Changing columns 3 to 8 into boolean
true_false={'Star Wars: Episode I The Phantom Menace':True,
            'Star Wars: Episode VI Return of the Jedi':True, 
            'Star Wars: Episode II Attack of the Clones': True, 
            'Star Wars: Episode III Revenge of the Sith': True , 
            'Star Wars: Episode IV A New Hope':True, 
            'Star Wars: Episode V The Empire Strikes Back':True, 
            np.nan:False }
for col in star_wars.columns[3:9]:
    star_wars[col]=star_wars[col].map(true_false)
    
    


#Changing the column names to a more readable form
star_wars = star_wars.rename(columns={
    "Which of the following Star Wars films have you seen? Please select all that apply.": "seen_1",
    "Unnamed: 4": "seen_2",
    "Unnamed: 5": "seen_3",
    "Unnamed: 6": "seen_4",
    "Unnamed: 7": "seen_5",
    "Unnamed: 8": "seen_6"
})

#Changing the column names to a more readable form

star_wars = star_wars.rename(columns={
        "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "ranking_1",
        "Unnamed: 10": "ranking_2",
        "Unnamed: 11": "ranking_3",
        "Unnamed: 12": "ranking_4",
        "Unnamed: 13": "ranking_5",
        "Unnamed: 14": "ranking_6"
        })




star_wars[star_wars.columns[9:15]] = star_wars[star_wars.columns[9:15]].astype(float)



star_wars[star_wars.columns[3:9]].info()

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.bar(range(6), star_wars[star_wars.columns[9:15]].mean())
plt.xlabel("Movies")
plt.ylabel("Score")
plt.show()


males=star_wars[star_wars['Gender']=='Male']
females=star_wars[star_wars['Gender']=='Female']


plt.bar(range(6),males[males.columns[9:15]].mean() )
plt.xlabel("Male")
plt.ylabel("Score")
plt.show()
plt.bar(range(6),females[females.columns[9:15]].mean() )
plt.xlabel("Female")
plt.ylabel("Score")
plt.show()






