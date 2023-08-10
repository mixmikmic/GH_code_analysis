import pandas as pd
import numpy as np
import re

#read in game information
steam = pd.read_csv("./steam_all/steam_all_last_attempt.csv")

steam1 = steam

steam1.shape

steam1.columns

steam1["no_user"][0]

steam1["no_user"] = steam1["no_user"].str.strip()

#grabbing percentage of positive reviews
steam1["percent"] = steam1["no_user"].str.extract("- (\d{2,3})\% .+")

steam1["percent"] = pd.to_numeric(steam1["percent"])

#grabbing number of reviews
steam1["reivew_users"]=pd.to_numeric(steam1["no_user"].str.extract("- \d{2,3}\% of the (.+) user .+").str.replace(",",""))

#construct a keyword list without extra spaces and lines
for i in range(steam1.shape[0]):
    steam1.loc[i,"keywords_list"] = ', '.join(map(lambda x: x.strip(),str(steam1["keywords"][i]).split(",")))

steam1["type"].value_counts()

#checking if original game differs from title - these should match with DLC and other type
for i in range(steam1.shape[0]):
    if steam1.loc[i,"title"] != steam1.loc[i,"mother_node"]:
        steam1.loc[i,"sub"] = 1

#identifying listing that are not downloadable content but has a different original game
for i in range(steam1.shape[0]):
    if (steam1.loc[i,"sub"] == 1)&(steam1.loc[i,"type"]!="Downloadable Content"):
        steam1.loc[i,"dlc_notsub"] = 1

#review user rating
steam1["user_rating"].value_counts()

#assign not-found to NaN
steam1.replace("not-found",np.NaN, inplace=True)

#extracting genres from a list that includes both publishers and producers
for i in range(steam1.shape[0]):
    steam1.loc[i,"genre_list"] = ", ".join(list(set(str(steam1["genre"][i]).split(","))^set(str(steam1["publish"][i]).split(","))))

#size of the dataset
steam1.shape

steam1["type"].value_counts()

#the final dataset should only consist of original games - soundtracks and downloadable content information
# will be added to the original game
a = steam1[(steam1["type"]=="None")].groupby("mother_node").first().reset_index()

a.shape

a.rename(columns={"title":"title_steam"},inplace=True)

#creating a database to count the number of DLCs
downloadable = steam1.loc[steam1["type"]=="Downloadable Content"]

downloadable['type'].value_counts()

downloadable.columns

dlc_count = downloadable[["title","mother_node"]].groupby("mother_node").agg(["count"])

dlc_count1 = dlc_count.reset_index()

#making sure that there is no repating values
dlc_count1.columns = dlc_count1.columns.get_level_values(0)

dlc_count1.rename(columns={'title':'no_dlc'}, inplace=True)
dlc_count1.rename(columns={'mother_node':'mother_dlc'}, inplace=True)

dlc_count1.columns

dlc_count1.shape

a.columns

#adding number of dlc in the 
games_dlc=pd.merge(a, dlc_count1, how="left", left_on = "mother_node", right_on = "mother_dlc")

for i in range(games_dlc.shape[0]):
    if pd.isnull(games_dlc.loc[i, "no_dlc"])==True:
        games_dlc.loc[i,"no_dlc"] = 0

games_dlc.loc[games_dlc["no_dlc"]>=1,"dlc_av"] = 1
games_dlc.loc[games_dlc["no_dlc"]==0,"dlc_av"] = 0

games_dlc["dlc_av"].value_counts()

games_dlc.columns

games_dlc.shape

#reading in data on top seller ranking and price
steam_rank = pd.read_csv("./ranking and price/ranking_withprice.csv")

steam_rank.shape

# revmoing duplicate records - I found out that the original steam listing consisted of repeating records of the same game
# while the ranking is different, the price is the same.
steam_rank_no = steam_rank[["title","rank_no"]].sort_values(by = "rank_no").groupby(by="title").first().reset_index()

steam_rank_no.head(3)

steam_rank_no.loc[0,"title"]=""

steam_rank_no.shape

# clearning up price data
steam_rank["price"] = steam_rank["price"].str.strip()

for i in range(len(steam_rank)):
    if type(steam_rank.loc[i,"price"]) is str:
        if steam_rank.loc[i,"price"] == "Free":
            steam_rank.loc[i,"price"] = 0
        else:
            steam_rank.loc[i,"price"] = float(steam_rank.loc[i,"price"])
    else:
        steam_rank.loc[i,"price"] = np.NaN

steam_rank[["title","price"]].head(5)

sum(pd.isnull(steam_rank["price"]))

steam_rank["price"]=steam_rank["price"].astype(float)

#removing duplicates
steam_price = steam_rank[["title","price"]].groupby("title").agg("mean")

steam_price = steam_price.reset_index()

steam_price.shape

#creating a df with unique price and rank
steam_price_rank = pd.merge(steam_price, steam_rank_no,how = "inner",left_on = "title", right_on= "title")

steam_price_rank.shape

# extracting game and original game relationship from the game information df
steam_name = steam1[["mother_node","title"]]

steam_name = steam_name.drop_duplicates()

steam_name.shape

steam_name.columns

steam_price_rank.columns

#merge rank and price with game and oritinal game mapping 
ranking = pd.merge(steam_name, steam_price_rank, how="inner",left_on="title",right_on="title")

ranking.shape

ranking = ranking.sort_values("rank_no")

#removing duplicate games 
ranking_mother = ranking.groupby("mother_node").first().reset_index()

ranking_mother.head(3)

ranking_mother.shape

ranking_mother.columns

games_dlc.columns

ranking_mother.rename(columns={'mother_node':'mother_rank','title':'title_rank'}, inplace=True)

#merging information back to the unique game df
rank = pd.merge(games_dlc, ranking_mother, how = "inner", left_on = "mother_node", right_on = "mother_rank")

rank.shape

sum(pd.isnull(rank["rank_no"]))

sum(pd.isnull(rank["Rdate"]))

rank.columns

#adding information regarding current player no, and all time high
steam_player = pd.read_csv("./player_count/users1.csv")

steam_player.shape

steam_player.columns

steam_app = steam1[["mother_node","title","app_id"]]

steam_app = steam_app.drop_duplicates()

steam_app.head(5)

#adding original game and game mapping information
name_player = pd.merge(steam_player, steam_app, how="inner",left_on = "app_id",right_on = "app_id")

name_player.columns

name_player.shape

#converting player stats into numeric
name_player["all_time"] = pd.to_numeric(name_player["all_time"].str.replace(",",""))

for i in range(len(name_player)):
    name_player.loc[i,"current"] = float(name_player.loc[i,"current"])

name_player["day"] = pd.to_numeric(name_player["day"].str.replace(",",""))

#creating a mean number of player for each original game
name_player_avg = name_player.groupby('mother_node').agg("mean").reset_index()

name_player_avg.head(5)

cols = pd.MultiIndex.from_tuples([("mean", "mother_node"), ("mean", "current"),("mean","all_time"),("mean","app_id"),("mean","24_peak")])

name_player_avg.columns = cols

name_player_avg.columns = name_player_avg.columns.droplevel()

name_player_avg.head(5)

name_player_avg.columns

name_player_avg  = name_player_avg[["mother_node","current","all_time","24_peak"]]

#adding inforamtion back together
final_player = pd.merge(rank, name_player_avg, how = "inner",left_on = "mother_node", right_on = "mother_node")

final_player.shape

final_player.columns

final_player.head(2)

sum(pd.isnull(final_player["Rdate"]))

final_player["str_date"] = final_player["Rdate"].str.contains(",")

#cleaning up the release date format
for i in range(len(final_player)):
    if final_player.loc[i, "str_date"] is True:
        final_player.loc[i,"DATE"] = pd.to_datetime(final_player.loc[i,"Rdate"], format = "%b %d, %Y")
    else:
        final_player.loc[i,"DATE"] = pd.to_datetime(final_player.loc[i,"Rdate"], format = "%b %Y")

final_player.head(2)

final_player.columns

import datetime
datetime.date.today()

#computing time since game released
final_player["time_diff"] = (datetime.date.today() - final_player["DATE"])

final_player.head(1)

final_player.columns.values

for i in range(len(final_player)):
    final_player.loc[i,"YEAR"] = final_player.loc[i,"DATE"].year

#selecting cleaned up variables
final_file = final_player[["mother_node","title_steam","app_id","user_rating","game_spcs","percent","reivew_users","keywords_list","genre_list","no_dlc","dlc_av","rank_no","current","all_time","24_peak","time_diff","YEAR","price","DATE"]]

final_file.head(1)

final_file.shape

keyword_dict = dict()

final_file.columns.values

final_file.head(1)

#creating a unique keyword list that does not include genre words
def unique(x, y):
    a = x.split(', ') 
    b = y.split(', ')
    unique = list(set(a)-set(b))
    unique = ", ".join(unique)
    return unique
final_file.loc[:,"unique"] = np.vectorize(unique)(final_file["keywords_list"], final_file["genre_list"])

final_file.unique.head(1)

#count top keywords
keyword_dict= dict()
for i in range(len(final_file)):
    a = final_file.loc[i,"unique"].split(", ")
    for i in a:
        if i not in keyword_dict:
            keyword_dict[i] = 1
        if i in keyword_dict:
            keyword_dict[i] = keyword_dict[i] + 1

key = pd.DataFrame(keyword_dict.items(),columns = ["keywords","count"]).sort_values("count",ascending= False)

key.head(30)

genre_dict= dict()
for i in range(len(final_file)):
    a = final_file.loc[i,"genre_list"].split(", ")
    for i in a:
        if i not in genre_dict:
            genre_dict[i] = 1
        if i in genre_dict:
            genre_dict[i] = genre_dict[i] + 1

c = pd.DataFrame(genre_dict.items(),columns = ["genre","count"]).sort_values("count",ascending= False)

c[c["count"] > 100]

#creating dummy variable for top genres
def genre_recode(x,text):
    a = x.split(', ')
    if text in a:
        return 1
    else:
        return 0
final_file.loc[:,"Indie"] = final_file["genre_list"].apply(lambda dic: genre_recode(dic, 'Indie'))
final_file.loc[:,"Action_g"] = final_file["genre_list"].apply(lambda dic: genre_recode(dic, 'Action'))
final_file.loc[:,"Adventure_g"] = final_file["genre_list"].apply(lambda dic: genre_recode(dic, 'Adventure'))
final_file.loc[:,"Strategy"] = final_file["genre_list"].apply(lambda dic: genre_recode(dic, 'Strategy'))
final_file.loc[:,"Simulation"] = final_file["genre_list"].apply(lambda dic: genre_recode(dic, 'Simulation'))
final_file.loc[:,"RPG"] = final_file["genre_list"].apply(lambda dic: genre_recode(dic, 'RPG'))
final_file.loc[:,"Casual"] = final_file["genre_list"].apply(lambda dic: genre_recode(dic, 'Casual'))
final_file.loc[:,"Early Access"] = final_file["genre_list"].apply(lambda dic: genre_recode(dic, 'Early Access'))
final_file.loc[:,"Sports"] = final_file["genre_list"].apply(lambda dic: genre_recode(dic, 'Sports'))
final_file.loc[:,"Violent"] = final_file["genre_list"].apply(lambda dic: genre_recode(dic, 'Violent'))
final_file.loc[:,"Racing"] = final_file["genre_list"].apply(lambda dic: genre_recode(dic, 'Racing'))

# checking if if-statements worked
final_file.head(1)[["genre_list","Indie","Adventure_g","Racing","RPG","Action_g","Violent"]]

np.sum(final_file.Indie)

#creating dummy variables for top keywords
def unique_recode(x,text):
    a = x.split(', ')
    if text in a:
        return 1
    else:
        return 0
final_file.loc[:,"Singleplayer"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Singleplayer'))
final_file.loc[:,"Multiplayer"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Multiplayer'))
final_file.loc[:,"Great Soundtrack"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Great Soundtrack'))
final_file.loc[:,"Atmospheric"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Atmospheric'))
final_file.loc[:,"Story Rich"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Story Rich'))
final_file.loc[:,"Open World"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Open World'))
final_file.loc[:,"2D"] = final_file["unique"].apply(lambda dic: unique_recode(dic, '2D'))
final_file.loc[:,"Co-op"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Co-op'))
final_file.loc[:,"Sci-fi"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Sci-fi'))
final_file.loc[:,"Adventure_k"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Adventure'))
final_file.loc[:,"Fantasy"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Fantasy'))
final_file.loc[:,"Puzzle"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Puzzle'))
final_file.loc[:,"First-Person"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'First-Person'))
final_file.loc[:,"Shooter"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Shooter'))
final_file.loc[:,"Difficult"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Difficult'))
final_file.loc[:,"Funny"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Funny'))
final_file.loc[:,"Classic"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Classic'))
final_file.loc[:,"Sandbox"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Sandbox'))
final_file.loc[:,"Horror"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Horror'))
final_file.loc[:,"Female Protagonist"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Female Protagonist'))
final_file.loc[:,"FPS"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'FPS'))
final_file.loc[:,"Pixel Graphics"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Pixel Graphics'))
final_file.loc[:,"Comedy"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Comedy'))
final_file.loc[:,"Survival"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Survival'))
final_file.loc[:,"Third Person"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Third Person'))
final_file.loc[:,"Anime"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Anime'))
final_file.loc[:,"Platformer"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Platformer'))
final_file.loc[:,"Action_k"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Action'))
final_file.loc[:,"Turn-Based"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Turn-Based'))
final_file.loc[:,"Exploration"] = final_file["unique"].apply(lambda dic: unique_recode(dic, 'Exploration'))

final_file.head(6)[["unique","Singleplayer","Anime","Fantasy"]]

np.sum(final_file["Singleplayer"])

final_file.columns.values

for i in range(len(final_file)):
    a = final_file.loc[i,"keywords_list"].split(", ")
    final_file.loc[i,"total_key"] = len(a)

final_file.head(2)

#calculating agreement
def agreement(x, y):
    a = x.split(', ') 
    b = y.split(', ')
    total = b
    agree = list(set(a).intersection(b))
    agreement = float(len(agree))/float(len(total))
    return agreement
final_file.loc[:,"agreement"] = np.vectorize(agreement)(final_file["keywords_list"], 
                                                        final_file["genre_list"])

final_file.head(2)

final_file["agreement"].describe()

final_file.loc[:,"retention"] = final_file["current"]/final_file["all_time"]*100

final_file.sort_values("retention",ascending = False)

#save final data
#final_file.to_csv("./cleaned_data.csv")

final_file = pd.read_csv("./cleaned_data.csv")

del final_file["Unnamed: 0"]

#import graphical packages
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import seaborn as sns



final_file.columns.values

#genre correlation
df = final_file[["Indie","Action_g","Adventure_g","Strategy","Simulation","RPG","Casual","Early Access","Sports","Violent","Racing"]]

#getting correlation (matthews) for binary, ORs, overlap%, and p-value
import scipy
from sklearn import metrics
def overlap(x, y, data):
    a = pd.crosstab(df[x] > 0, df[y] > 0)
    print a
    print "overlap %: " + str(round((float(a[1][1])/float(np.sum(a[0]+a[1])))*100,2))+"%"
    print "OR: " + str(round(float(a[0][0] * a[1][1])/float(a[0][1]*a[1][0]),2))
    b = scipy.stats.chi2_contingency(a)
    print b[1]
    print round(sklearn.metrics.matthews_corrcoef(df[x],df[y],sample_weight=None),2)

def ORs(x,y,data):
    a = pd.crosstab(data[x] > 0, data[y] > 0)
    return round(float(a[0][0] * a[1][1])/float(a[0][1]*a[1][0]),2)

col_name = ["Indie","Action_g","Adventure_g","Strategy","Simulation","RPG","Casual","Early Access","Sports",
            "Violent","Racing"]
col_name[0]

import sklearn
a = df.columns.values
b = a
for i in range(len(a)):
    j = i + 1 
    while j < len(b):
        overlap(a[i],a[j],df)
        j = j + 1 
        print "="*40

new = pd.DataFrame(columns = col_name, index = col_name )
a = df.columns.values
b = a
for i in range(len(a)):
    j = i
    new.loc[col_name[i],col_name[j]]=1
    j = i + 1 
    while j < len(b):
        new.loc[col_name[i],col_name[j]] = ORs(col_name[i],col_name[j],df)
        j = j + 1 

for i in range(len(a)):
    j = i
    new.loc[col_name[i],col_name[j]]=1
    j = i + 1 
    while j < len(b):
        new.loc[col_name[j],col_name[i]] = ORs(col_name[j],col_name[i],df)
        j = j + 1 

new

greater = 0
for i in range(len(a)):
    j = i + 1 
    while j < len(b):
        if new.loc[col_name[i],col_name[j]] > 1:
            greater = greater + 1
        j = j + 1 
float(greater)/(float(sum(range(1,12)))-11)

(float(sum(range(1,12)))-11)

less = 0
for i in range(len(a)):
    j = i + 1 
    while j < len(b):
        if new.loc[col_name[i],col_name[j]] < 1:
            less = less + 1
        j = j + 1 
float(less)/(float(sum(range(1,12)))-11)

final_file.loc[(final_file.Strategy == 1) & (final_file.Action_g == 1)].sort_values(by = "all_time")

df.corr()

#generating a correlation plot
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 220, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.4,
            square=True,
            linewidths=.5, cbar_kws={"shrink": .3}, ax=ax)
#none of the correlation nor the overlapping is very sigifnicant
sns.set_style("dark")

#matplot genre count
data = c[c['count']>100].sort_values(by = "count",ascending=False)
ax = sns.barplot(y="genre", x= "count", data=data,color="darkcyan",orient="h")
plt.ylabel('Company-defined Genre')
plt.xlabel('Counts')
sns.set_style("whitegrid")

b

#import plotly settings
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
import plotly
plotly.tools.set_credentials_file(username='jasonchiu0803', api_key='Jk1S0TST6AbhYxuAnYPD')

#interactive word counts for genre
import plotly.plotly as py
import plotly.graph_objs as go
subset_genre = b[b["count"]>100].sort_values(by = "count",ascending=True)
data = [go.Bar(
            x=subset_genre["count"],
            y=subset_genre["genre"],
            orientation = 'h')]
layout = go.Layout(
    title="Top Company Defined Genre",
    xaxis=dict(
        title="Counts"),
    yaxis=dict(
        title='Company Defined Genre'),autosize=True)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='horizontal-bar')

df_g = final_file[["Indie","Action_g","Adventure_g","Strategy","Simulation","RPG","Casual","Early Access",
                   "Sports","Violent","Racing","all_time"]]

genre_name = ["Indie","Action_g","Adventure_g","Strategy","Simulation","RPG","Casual","Early Access","Sports",
            "Violent","Racing"]

def compiling_data(x):
    boxplot_data = pd.DataFrame(columns = ["present","all_time","genre_title"])
    for i in range(len(x)):
        y = genre_name[i]
        new = df_g[[y,"all_time"]]
        new.loc[:,"genre_title"] = y
        new.columns = ["present","all_time","genre_title"]
        boxplot_data = pd.concat([boxplot_data, new])
    return boxplot_data

boxplot_data = compiling_data(genre_name)

boxplot_data.head(2)

boxplot_data.shape

# Code for plotly plot
'''import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

Not_Action = go.Box(y = list(final_file['all_time'][final_file['Action_g']==0]),
                    boxpoints = False)
Action = go.Box(y = list(final_file['all_time'][final_file['Action_g']==1]),
                boxpoints = False)
data = [Not_Action, Action]
layout = go.Layout(yaxis=dict(
        range=[0, 4000]
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='axes-range-manual')'''

subbox = boxplot_data[boxplot_data["genre_title"].isin(["Indie","Casual","Action_g","RPG"])]

subbox.columns

g = sns.FacetGrid(subbox, col="genre_title", size=4, aspect=.5, col_order = ["Indie","Casual","Action_g","RPG"])
g.map(sns.boxplot, "present", "all_time",showfliers=False);
axes = g.axes
axes[0,0].set_ylim(0,4000)
axes[0,1].set_ylim(0,4000)
axes[0,2].set_ylim(0,4000)
axes[0,3].set_ylim(0,4000)
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Genre Group Comparison - All Time Player No.')

import scipy
def ttest_al(x):
    print final_file[["all_time",x]].groupby(x).agg("mean")
    print scipy.stats.ttest_ind(final_file["all_time"][final_file[x]==0],final_file["all_time"][final_file[x]==1])[1]

for i in range(len(genre_name)):
    ttest_al(genre_name[i])
    print "="*40

def compiling_data_r(x):
    boxplot_data = pd.DataFrame(columns = ["present","retention","genre_title"])
    for i in range(len(x)):
        y = genre_name[i]
        new = df_g[[y,"retention"]]
        new.loc[:,"genre_title"] = y
        new.columns = ["present","retention","genre_title"]
        boxplot_data = pd.concat([boxplot_data, new])
    return boxplot_data

boxplot_data = compiling_data_r(genre_name)

import scipy
def ttest_r(x):
    print final_file[["retention",x]].groupby(x).agg("mean")
    print scipy.stats.ttest_ind(final_file["retention"][final_file[x]==0],final_file["retention"][final_file[x]==1])[1]

for i in range(len(genre_name)):
    ttest_r(genre_name[i])
    print "="*40

subbox = boxplot_data[boxplot_data["genre_title"].isin(["Indie","Casual","Action_g","RPG"])]

#return to do analysis on retention

#correlation between keywords
df_k = final_file[["Singleplayer","Multiplayer","Great Soundtrack","Atmospheric","Story Rich",
                 "Open World","2D","Co-op","Sci-fi","Fantasy","Adventure_k","Puzzle","First-Person","Shooter",
                 "Difficult","Funny","Classic","Sandbox","Horror","Female Protagonist",
                 "FPS","Pixel Graphics","Comedy","Survival","Third Person","Anime","Platformer","Action_k","Turn-Based",
                "Exploration"]]

col_name_k = df_k.columns.values
new_k = pd.DataFrame(columns = col_name_k, index = col_name_k)

a_k = col_name_k
for i in range(len(a_k)):
    j = i
    new_k.loc[col_name_k[i],col_name_k[j]]=1
    j = i + 1 
    while j < len(a_k):
        new_k.loc[col_name_k[i],col_name_k[j]] = ORs(col_name_k[i],col_name_k[j],df_k)
        j = j + 1 

for i in range(len(a_k)):
    j = i
    new_k.loc[col_name_k[i],col_name_k[j]]=1
    j = i + 1 
    while j < len(a_k):
        new_k.loc[col_name_k[j],col_name_k[i]] = ORs(col_name_k[j],col_name_k[i],df_k)
        j = j + 1 

pd.set_option('display.max_columns', 100)

new_k

df_k.corr()

#generating a correlation plot
corr = df_k.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 220, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.4,
            square=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
#none of the correlation nor the overlapping is very sigifnicant
sns.set_style("whitegrid")

greater_k = 0
for i in range(len(a_k)):
    j = i + 1 
    while j < len(a_k):
        if new_k.loc[col_name_k[i],col_name_k[j]] > 1:
            greater_k = greater_k + 1
        j = j + 1 
greater_k_per = float(greater_k)/(float(sum(range(1,30)))-28)

greater_k_per

(float(sum(range(1,29)))-28)

less_k = 0
for i in range(len(a_k)):
    j = i + 1 
    while j < len(a_k):
        if new_k.loc[col_name_k[i],col_name_k[j]] < 1:
            less_k = less_k + 1
        j = j + 1 
less_k_per = float(less_k)/(float(sum(range(1,30)))-28)

less_k_per

test_list = ["Singleplayer","Multiplayer","Great Soundtrack","Atmospheric","Story Rich","Open World","2D","Co-op",
             "Sci-fi","Adventure_k","Fantasy","Puzzle","First-Person","Shooter","Difficult","Funny","Classic",
             "Sandbox","Horror","Female Protagonist","FPS","Pixel Graphics","Comedy","Survival","Third Person",
             "Anime","Platformer","Action_k","Turn-Based","Exploration"]

len(test_list)

#insepecting all_time players/ 2 sample t-test/ p-value
import scipy
def ttest_al(x):
    print final_file[["all_time",x]].groupby(x).agg("mean")
    print scipy.stats.ttest_ind(final_file["all_time"][final_file[x]==0],final_file["all_time"][final_file[x]==1])[1]

for i in range(len(test_list)):
    ttest_al(test_list[i])
    print "="*50

#insepecting retention relationship
def ttest_re(x):
    print final_file[["retention",x]].groupby(x).agg("mean")
    print scipy.stats.ttest_ind(final_file["retention"][final_file[x]==0],final_file["retention"][final_file[x]==1])[1]

#matplot word count
key[key["count"]>230].sort_values(by = "count",ascending=True).plot.barh(x="keywords")
fig_size = plt.rcParams["figure.figsize"]
print "Current size:", fig_size
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
plt.ylabel('User-defined keywords')

#interactive word counts for keywords
import plotly.plotly as py
import plotly.graph_objs as go
subset_key = key[key["count"]>230].sort_values(by="count")
data = [go.Bar(
            x=subset_key["count"],
            y=subset_key["keywords"],
            orientation = 'h')]
layout = go.Layout(
    title="User-defined Keywords",
    xaxis=dict(
        title="Counts"),
    yaxis=dict(
        title='User-defined Keywrods'),autosize=True)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='horizontal-bar')

test_list = ["Singleplayer","Multiplayer","Great Soundtrack","Atmospheric","Story Rich","Open World","2D","Co-op",
             "Sci-fi","Adventure_k","Fantasy","Puzzle","First-Person","Shooter","Difficult","Funny","Classic",
             "Sandbox","Horror","Female Protagonist","FPS","Pixel Graphics","Comedy","Survival","Third Person",
             "Anime","Platformer","Action_k","Turn-Based","Exploration"]



df_key = final_file[["Singleplayer","Multiplayer","Great Soundtrack","Atmospheric","Story Rich","Open World","2D","Co-op",
             "Sci-fi","Adventure_k","Fantasy","Puzzle","First-Person","Shooter","Difficult","Funny","Classic",
             "Sandbox","Horror","Female Protagonist","FPS","Pixel Graphics","Comedy","Survival","Third Person",
             "Anime","Platformer","Action_k","Turn-Based","Exploration","all_time"]]

def compiling_data_k(x):
    boxplot_data = pd.DataFrame(columns = ["present","all_time","keywords_title"])
    for i in range(len(x)):
        y = test_list[i]
        new = df_key[[y,"all_time"]]
        new.loc[:,"keywords_title"] = y
        new.columns = ["present","all_time","keywords_title"]
        boxplot_data = pd.concat([boxplot_data, new])
    return boxplot_data

boxplot_k = compiling_data_k(test_list)

positive = boxplot_k[boxplot_k["keywords_title"].isin(["Singleplayer","Atmospheric","Co-op","Sci-fi",
                                                       "Sandbox","FPS","Third Person","Exploration"])]

g = sns.FacetGrid(positive, col="keywords_title", size=4, aspect=.5, col_order = ['Co-op','Atmospheric',
                                                                                  'Sci-fi','Sandbox','FPS',
                                                                                  'Third Person'])
g.map(sns.boxplot, "present", "all_time",showfliers=False, color="darkcyan");
axes = g.axes
axes[0,0].set_ylim(0,5000)
axes[0,1].set_ylim(0,5000)
axes[0,2].set_ylim(0,5000)
axes[0,3].set_ylim(0,14000)
plt.subplots_adjust(top=0.85)
g.fig.suptitle('User-defined Keyword Group Comparison - All Time Player No.')
sns.set_style("whitegrid")

negative = boxplot_k[boxplot_k["keywords_title"].isin(["Puzzle","Anime"])]

g = sns.FacetGrid(negative, col="keywords_title", size=4, aspect=.5, col_order = ['Puzzle','Anime'])
g.map(sns.boxplot, "present", "all_time",showfliers=False, color = "darkcyan");
axes = g.axes
axes[0,0].set_ylim(0,2500)
axes[0,1].set_ylim(0,2500)
sns.set_style("whitegrid")
plt.subplots_adjust(top=0.85)
g.fig.suptitle('User-defined Keyword Group Comparison - All Time Player No.')

# all_time
import seaborn as sns
sns.distplot(final_file['all_time'], bins=1500, kde=False, rug=False,color = "darkcyan");
plt.xlim(0, 10000)
sns.set_style("whitegrid")
plt.ylabel('Count')
plt.xlabel('Highest Volume of Players')

final_file['all_time'].describe()

final_file[['mother_node','all_time']].sort_values(by = "all_time", ascending = False).head(10).reset_index()

final_file[['mother_node','all_time']][final_file["all_time"]<100].shape

final_file.columns.values

final_file[["YEAR","mother_node"]].groupby("YEAR").agg("count").reset_index().sort_values("YEAR",ascending = False)

final_file[final_file["YEAR"]==1987.0]

#plotting number of all_time users with released years
ax = sns.regplot(x="YEAR", y="all_time", data=final_file, ci=0, color = "darkcyan")
plt.xlim(1990, 2018)
sns.plt.suptitle('Game Released Year and Highest Volumn of Players')
ax.set(xlabel='Released Year', ylabel='Highest Volumn of Players')
sns.set_style("whitegrid")

year_count = final_file[["YEAR","mother_node"]].groupby("YEAR").agg("count").reset_index()

year_count.columns

year_count

pd.options.display.float_format = '{:f}'.format

year_count.YEAR.values

year_label = map(lambda x: str(x)[0:4], year_count.YEAR.values)

sns.set_style("whitegrid")
ax = sns.pointplot(x="YEAR", y="mother_node", data=year_count, color = "darkcyan")
ax.set_xticklabels(labels = year_label, rotation=40)
plt.ylim(0, 1000)
ax.set(xlabel='Released Year', ylabel='Counts')
ax.set_title('No of Games by Released Year')

min(final_file["all_time"])

scipy.stats.ttest_ind(final_file["all_time"][final_file["dlc_av"]==0],final_file["dlc_av"][final_file[x]==1])

final_file[["all_time",'dlc_av']].groupby("dlc_av").agg("mean")

#with zeros
sns.distplot(final_file['no_dlc'], bins=200, kde=False, rug=False,color = "darkcyan");
sns.set_style("whitegrid")
plt.ylabel('Count')
plt.xlim(0, 10)
plt.xlabel('No. of DLCs')
plt.rcParams['figure.figsize']=(6,6)

ax = sns.boxplot(x="dlc_av", y="all_time", data=final_file, showfliers=False)
plt.ylim(0, 20000)
ax.set(xlabel='Downloadable Content Avaliability', ylabel="Highest Volume of Players")

final_file.no_dlc.describe()

dlc = final_file[final_file["no_dlc"]>0]

# only games with dlc
sns.set_style("whitegrid")
ax = sns.distplot(dlc["no_dlc"], bins=80, kde=False, rug=False, color = "darkcyan");
ax.set(xlabel='No. of DLCs', ylabel='Counts')
plt.xlim(0, 50)
ax.set_title('DLCs')





456./(2624+456)

ax = sns.regplot(x="no_dlc", y="all_time", data=final_file, x_jitter = 0.3,color = "darkcyan",ci=0)
plt.ylim(0, 150000)
plt.xlim(0, 50)
sns.set_style("whitegrid")

final_file["no_dlc"].describe()

# visualization for price
ax = sns.regplot(x="price", y="all_time", data=final_file,x_jitter=.5, color = "darkcyan", ci = 0)
plt.ylim(0, 300000)
plt.xlim(0, 150)
sns.set_style("whitegrid")

final_file[final_file["price"]>150]

sns.set_style("whitegrid")
ax = sns.distplot(final_file['price'], bins=80, kde=False, rug=False, color = "darkcyan");
plt.ylim(0, 1000)
plt.xlim(0, 100)
ax.set(xlabel='Price ($)', ylabel='Counts')
ax.set_title('Game Prices')

final_file["price"].describe()

final_file[final_file["price"]>200]

final_file.columns.values

con_df = final_file[["price","no_dlc","YEAR","agreement","total_key","retention","rank_no","all_time"]]

corr = con_df.corr()

corr

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 220, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.4,
            square=True,
            linewidths=.5, cbar_kws={"shrink": .3}, ax=ax)
#none of the correlation nor the overlapping is very sigifnicant
sns.set_style("dark")

final_file.columns.values

final_file[final_file["Anime"]==1]



















