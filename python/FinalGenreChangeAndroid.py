import pandas as pd

df=pd.read_json(path_or_buf="clean_android_all.json")

df.columns

df["genre"].unique()

dfitunes=pd.read_json(path_or_buf="cleaned_itunes.json", typ="frame",lines=True)

dfitunes.columns

dfitunes["genre"].unique()

df["mod_genre"]=df["genre"]

df["mod_genre"].replace('Social','Social Networking',inplace=True)

df.head(10)

df["mod_genre"].replace('Brain & Puzzle','Games',inplace=True)

df["mod_genre"].replace('Media & Video','Photo & Video',inplace=True)

df["mod_genre"].replace('Personalization','Entertainment',inplace=True)

df["mod_genre"].replace('Photography','Photo & Video',inplace=True)

df["mod_genre"].replace('Arcade & Action','Games',inplace=True)

df["mod_genre"].replace('Communication','Social Networking',inplace=True)

df["mod_genre"].replace('Tools','Utilities',inplace=True)

df["mod_genre"].replace('Comics','Entertainment',inplace=True)

df["mod_genre"].replace('News & Magazines','News',inplace=True)

df["mod_genre"].replace('Travel & Local','Travel',inplace=True)

df["mod_genre"].replace('Music & Audio','Music',inplace=True)

df["mod_genre"].replace('Racing','Games',inplace=True)

df["mod_genre"].replace('Casual','Games',inplace=True)

df["mod_genre"].replace('Transportation','Navigation',inplace=True)

df["mod_genre"].replace('Libraries & Demo','Entertainment',inplace=True)

df["mod_genre"].replace('Cards & Casino','Games',inplace=True)

df["mod_genre"].replace('Sports Games','Games',inplace=True)

len(df["genre"].unique())

df["genre"].unique()

len(df["mod_genre"].unique())

df.head(10)

dfitunes.columns

dfitunes["genre"].replace('Food & Drink','Lifestyle',inplace=True)

dfitunes["genre"].replace('Books','Books & Reference',inplace=True)

dfitunes["genre"].replace('Reference','Books & Reference',inplace=True)

dfitunes["genre"].unique()

df.to_json(path_or_buf="clean_android_allcolumns.json",orient="records",lines=True)

dftemp= pd.read_json(path_or_buf="clean_android_allcolumns.json", typ="frame", lines=True)

dftemp.head(10)

dftemp["WOM"] = 0
dftemp.head(10)

df_utilities = dftemp[dftemp.mod_genre == 'Utilities']

df_utilities.head(10)

df_utilities["WOM"] = df_utilities.downloads.mean()/df_utilities.all_rating_count.mean()

df_utilities.head(10)

dfratios

dfratios=pd.DataFrame(dftemp["mod_genre"])

dfratios.drop_duplicates(inplace=True)

dfratios

dfratios["WOM"]=0

dfratios

dfratios.iloc[10,1]=229.62

dfratios

df_finance = dftemp[dftemp.mod_genre == 'Finance']
df_games = dftemp[dftemp.mod_genre == 'Games']
df_education = dftemp[dftemp.mod_genre == 'Education']
df_books = dftemp[dftemp.mod_genre == 'Books & Reference']
df_lifestyle = dftemp[dftemp.mod_genre == 'Lifestyle']

df_finance.downloads.mean()/df_finance.all_rating_count.mean()

dfratios.iloc[17,1]=197.2

dfratios

df_games.downloads.mean()/df_games.all_rating_count.mean()

dfratios.iloc[3,1]=132.16

df_education.downloads.mean()/df_education.all_rating_count.mean()

dfratios.iloc[5,1]=324.74

df_books.downloads.mean()/df_books.all_rating_count.mean()

dfratios.iloc[8,1]=217.47

df_lifestyle.downloads.mean()/df_lifestyle.all_rating_count.mean()

dfratios.iloc[7,1]=287.84

df_music = dftemp[dftemp.mod_genre == 'Music']
df_socialnetworking = dftemp[dftemp.mod_genre == 'Social Networking']
df_news  = dftemp[dftemp.mod_genre == 'News']
df_entertainment = dftemp[dftemp.mod_genre == 'Entertainment']
df_photo = dftemp[dftemp.mod_genre == 'Photo & Video']
df_navigation = dftemp[dftemp.mod_genre == 'Navigation']
df_health = dftemp[dftemp.mod_genre == 'Health & Fitness']
df_sports = dftemp[dftemp.mod_genre == 'Sports']
df_business = dftemp[dftemp.mod_genre == 'Business']
df_productivity = dftemp[dftemp.mod_genre == 'Productivity']
df_travel = dftemp[dftemp.mod_genre == 'Travel']

df_medical = dftemp[dftemp.mod_genre == 'Medical']
df_weather = dftemp[dftemp.mod_genre == 'Weather']
df_shopping = dftemp[dftemp.mod_genre == 'Shopping']

dfratios.iloc[15,1] = df_music.downloads.mean()/df_music.all_rating_count.mean()
dfratios.iloc[0,1] = df_socialnetworking.downloads.mean()/df_socialnetworking.all_rating_count.mean()
dfratios.iloc[12,1] = df_news.downloads.mean()/df_news.all_rating_count.mean()
dfratios.iloc[1,1] = df_entertainment.downloads.mean()/df_entertainment.all_rating_count.mean()
dfratios.iloc[4,1] = df_photo.downloads.mean()/df_photo.all_rating_count.mean()
dfratios.iloc[18,1] = df_navigation.downloads.mean()/df_navigation.all_rating_count.mean()
dfratios.iloc[6,1] = df_health.downloads.mean()/df_health.all_rating_count.mean()
dfratios.iloc[16,1] = df_sports.downloads.mean()/df_sports.all_rating_count.mean()
dfratios.iloc[11,1] = df_business.downloads.mean()/df_business.all_rating_count.mean()
dfratios.iloc[9,1] = df_productivity.downloads.mean()/df_productivity.all_rating_count.mean()
dfratios.iloc[13,1] = df_travel.downloads.mean()/df_travel.all_rating_count.mean()
dfratios.iloc[14,1] = df_medical.downloads.mean()/df_medical.all_rating_count.mean()
dfratios.iloc[19,1] = df_weather.downloads.mean()/df_weather.all_rating_count.mean()
dfratios.iloc[2,1] = df_shopping.downloads.mean()/df_shopping.all_rating_count.mean()

dfratios

df_utilities.corr()

df_books.corr()

df_business.corr()

df_productivity.corr()

dfratios.to_csv("alldata_genrewom.csv")

dfratios["mod_genre"]

dftemp.to_csv(path_or_buf="clean_android_genre.csv", header=True, index=False, encoding="utf-8")

df_utilities.to_csv(path_or_buf="android_utilities.csv", header=True, index=False, encoding="utf-8")

