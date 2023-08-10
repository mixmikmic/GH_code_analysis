# <aghatpande> on 07-Mar-2017
# print all the outputs in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import pandas as pd
import numpy as np

class Jsonloads:
    def read_in_chunks(self, file_object, chunk_size=500000000):
        while True:
            data = file_object.read(chunk_size)
            if not data:
                break
            yield data

count = 1;
# <ghatpande> on 14-April-2017 for AppMeter
# Give the path of the large input JSON file
f = open('Original_Data_from_Luka.json')
for piece in Jsonloads().read_in_chunks(f):
    pathname = "part" + str(count) + ".csv"
    text_file = open(pathname, "w")
    text_file.write(piece)
    text_file.close()
    count = count + 1;
    print("\n ==========The new chunk is ======== \n")

print("Count is %d" %count);

df = pd.read_json("part1.json", typ="frame", lines=True)

df.head(2)
df.columns
len(df)

df1 = df[["all_rating", "all_rating_count", "app_name", "bundle_id", "content_rating", "description", "downloads", "file_size", "genre", "icon_url", "id", "price", "version", "whats_new", "status_unix_timestamp", "status_date"]]

df1.head(2)
len(df1)

df1.to_csv(path_or_buf="part1.csv", header=True, index=False, encoding="utf-8")

dftemp1 = pd.read_csv("part1.csv")
len(dftemp1)
dftemp1.head(2)
dftemp1.dtypes

df = pd.read_json("part2.json", typ="frame", lines=True)

df2 = df[["all_rating", "all_rating_count", "app_name", "bundle_id", "content_rating", "description", "downloads", "file_size", "genre", "icon_url", "id", "price", "version", "whats_new", "status_unix_timestamp", "status_date"]]

df2.to_csv(path_or_buf="part2.csv", header=True, index=False, encoding="utf-8")

dftemp2 = pd.read_csv("part2.csv")
len(dftemp2)
dftemp2.head(2)
dftemp2.dtypes

df = pd.read_json("part3.json", typ="frame", lines=True)

df3 = df[["all_rating", "all_rating_count", "app_name", "bundle_id", "content_rating", "description", "downloads", "file_size", "genre", "icon_url", "id", "price", "version", "whats_new", "status_unix_timestamp", "status_date"]]

df3.to_csv(path_or_buf="part3.csv", header=True, index=False, encoding="utf-8")

dftemp3 = pd.read_csv("part3.csv")
len(dftemp3)
dftemp3.head(2)
dftemp3.dtypes

df = pd.read_json("part4.json", typ="frame", lines=True)

df4 = df[["all_rating", "all_rating_count", "app_name", "bundle_id", "content_rating", "description", "downloads", "file_size", "genre", "icon_url", "id", "price", "version", "whats_new", "status_unix_timestamp", "status_date"]]

df4.to_csv(path_or_buf="part4.csv", header=True, index=False, encoding="utf-8")

dftemp4 = pd.read_csv("part4.csv")
len(dftemp4)
dftemp4.head(2)
dftemp4.dtypes

len(dftemp1)
len(dftemp2)
len(dftemp3)
len(dftemp4)

#Combine all the dataframes
dfcombined = dftemp1.append(dftemp2)
dfcombined = dfcombined.append(dftemp3)
dfcombined = dfcombined.append(dftemp4)
len(dfcombined)

# Remove all the entries with 0 raters
dfcombined = dfcombined[dfcombined.all_rating_count != 0]
len(dfcombined)

#Change price column

def removedollar(x):
    tempstring=str(x)
    if str(x)=="Free":
        tempstring=float("0")
    elif str(x).startswith("$"):
        tempstring1=str(x).lstrip("$")
        tempstring=float(tempstring1)
    return float(tempstring)

dfcombined["price1"]=dfcombined["price"].apply (lambda x: removedollar(str(x)))

len(dfcombined)

# Remove the NaN from the downloads and AppName columns
dfcombined = dfcombined[dfcombined.downloads.isnull() != True]
dfcombined = dfcombined[dfcombined.app_name.isnull() != True]

len(dfcombined)

#Make Sure none of the columns are NaN
pd.isnull(dfcombined).sum() > 0

# Change the downloads column
import string
def modifydownloads(x):
    tempstring = str(x)
    if str(x)=="1 - 5":
        tempstring = str(x).replace("1 - 5","3")
    elif str(x)=="5 - 10":
        tempstring = str(x).replace("5 - 10","8")
    elif str(x)=="10 - 50":
        tempstring = str(x).replace("10 - 50","30")
    elif str(x)=="50 - 100":
        tempstring = str(x).replace("50 - 100","75")
    elif str(x)=="100 - 500":
        tempstring = str(x).replace("100 - 500","300")
    elif str(x)=="500 - 1,000":
        tempstring = str(x).replace("500 - 1,000","750")
    elif str(x)=="1,000 - 5,000":
        tempstring = str(x).replace("1,000 - 5,000","3000")
    elif str(x)=="5,000 - 10,000":
        tempstring = str(x).replace("5,000 - 10,000","7500")
    elif str(x)=="10,000 - 50,000":
        tempstring = str(x).replace("10,000 - 50,000","30000")
    elif str(x)=="50,000 - 100,000":
        tempstring = str(x).replace("50,000 - 100,000","75000")
    elif str(x)=="100,000 - 500,000":
        tempstring = str(x).replace("100,000 - 500,000","300000")
    elif str(x)=="500,000 - 1,000,000":
        tempstring = str(x).replace("500,000 - 1,000,000","750000")
    elif str(x)=="1,000,000 - 5,000,000":
        tempstring = str(x).replace("1,000,000 - 5,000,000","3000000")
    elif str(x)=="5,000,000 - 10,000,000":
        tempstring = str(x).replace("5,000,000 - 10,000,000","7500000")
    elif str(x)=="10,000,000 - 50,000,000":
        tempstring = str(x).replace("10,000,000 - 50,000,000","30000000")
    elif str(x)=="50,000,000 - 100,000,000":
        tempstring = str(x).replace("50,000,000 - 100,000,000","75000000")
    elif str(x)=="100,000,000 - 500,000,000":
        tempstring = str(x).replace("100,000,000 - 500,000,000","300000000")
    else:
        tempstring = str(x).replace("","0")
    return int(tempstring)

dfcombined = dfcombined[dfcombined.downloads != 0]
dfcombined["downloads1"] = dfcombined["downloads"].apply(lambda x: modifydownloads(str(x)))

len(dfcombined)
dfcombined.head(2)

# Change the sile_size column

def changeFileSize(x):
    tempstring = str(x)
    tempstring = tempstring.replace(",","")
    if str(x).endswith("k"):
        tempstring = tempstring.rstrip("k") 
        tempfloat = float(tempstring)
        return float(tempfloat/1000)
    if str(x).endswith("M"):
        tempstring = tempstring.rstrip("M")
        return float(tempstring)
    else:
        return int(0.0)

dfcombined["file_size1"] = dfcombined["file_size"].apply(lambda x: changeFileSize(str(x)))

len(dfcombined)
dfcombined.head(2)

# Remove the old columns.
dfcombined.drop('price', axis=1, inplace=True)

dfcombined.drop('downloads', axis=1, inplace=True)

dfcombined.drop('file_size', axis=1, inplace=True)

dfcombined.head(2)

#Rename the columns properly
dfcombined.rename(columns={"downloads1": "downloads", "file_size1": "file_size", "price1":"price"},inplace=True)

# Combine the genre's into logical group of reduced number of genre's
dfcombined["mod_genre"]= dfcombined["genre"]
dfcombined["mod_genre"].replace('Social','Social Networking',inplace=True)
dfcombined["mod_genre"].replace('Brain & Puzzle','Games',inplace=True)
dfcombined["mod_genre"].replace('Media & Video','Photo & Video',inplace=True)
dfcombined["mod_genre"].replace('Personalization','Entertainment',inplace=True)
dfcombined["mod_genre"].replace('Photography','Photo & Video',inplace=True)
dfcombined["mod_genre"].replace('Arcade & Action','Games',inplace=True)
dfcombined["mod_genre"].replace('Communication','Social Networking',inplace=True)
dfcombined["mod_genre"].replace('Tools','Utilities',inplace=True)
dfcombined["mod_genre"].replace('Comics','Entertainment',inplace=True)
dfcombined["mod_genre"].replace('News & Magazines','News',inplace=True)
dfcombined["mod_genre"].replace('Travel & Local','Travel',inplace=True)
dfcombined["mod_genre"].replace('Music & Audio','Music',inplace=True)
dfcombined["mod_genre"].replace('Racing','Games',inplace=True)
dfcombined["mod_genre"].replace('Casual','Games',inplace=True)
dfcombined["mod_genre"].replace('Transportation','Navigation',inplace=True)
dfcombined["mod_genre"].replace('Libraries & Demo','Entertainment',inplace=True)
dfcombined["mod_genre"].replace('Cards & Casino','Games',inplace=True)
dfcombined["mod_genre"].replace('Sports Games','Games',inplace=True)

len(dfcombined.genre.unique())
len(dfcombined.mod_genre.unique())
len(dfcombined)

# Remove the special characters from app_name column
dfcombined["app_name"].replace(to_replace = '[\!\.\+<>,@#$%^&\*()\-=\+\~]+',value = '', regex = True, inplace=True)

# Remove Non Ascii codes from the app_name column
dfcombined["app_name"].replace(to_replace = '[^\x00-\x7F]+',value = '', regex = True, inplace=True)

dfcombined.head(2)

# Save the dataframe as a csv file
dfcombined.to_csv("cleaned-android-file.csv", header=True, index=False, encoding="utf-8")



