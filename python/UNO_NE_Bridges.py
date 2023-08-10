import pymongo
from pymongo import MongoClient
client = MongoClient("mongodb://bridges:readonly@nbi-mongo.admin/bridge")
db = client.bridge
collection = db["bridges"]
print("Bridge Records in DB: ", collection.count())

structureID = "C003211015"

import pandas as pd
import re
pattern = re.compile(r'\s*' + re.escape(structureID) + r'.*', re.I)
records = collection.find({"structureNumber": {"$regex": pattern}})

dataframe = pd.DataFrame(list(records))
pd.set_option('display.max_columns', None)
print("# of bridges with " + structureID + " in Structure ID: " + str(len(dataframe.index)))

dataframe

get_ipython().run_line_magic('matplotlib', 'inline')

temp = dataframe.groupby(['year'])['deck'].max().reset_index()
temp.loc[temp['deck'] == 'N', 'deck'] = 10
temp.loc[temp['deck'] == '', 'deck'] = 0
temp['deck'] = temp['deck'].astype(int)
temp.plot(y = 'deck', x = 'year', marker = '.', title = "Deck Rating vs. Years")

temp = dataframe.groupby(['year'])['superstructure'].max().reset_index()
temp.loc[temp['superstructure'] == 'N', 'superstructure'] = 10
temp.loc[temp['superstructure'] == '', 'superstructure'] = 0
temp['superstructure'] = temp['superstructure'].astype(int)
temp.plot(y = 'superstructure', x = 'year', marker = '.', title = "Superstructure Rating vs. Years")

temp = dataframe.groupby(['year'])['yearBuilt'].max().reset_index()
temp['yearBuilt'] = temp['yearBuilt'].astype(int)
temp.plot(y='yearBuilt', x='year', marker='.', title = "Change in Year Built Value over Years")

temp = dataframe.groupby(['year'])['lengthOfStructureImprovement'].max().reset_index()
temp['lengthOfStructureImprovement'] = temp['lengthOfStructureImprovement'].astype(float)
temp.plot(y = 'lengthOfStructureImprovement', x = 'year', marker = '.', title = "Length Of Structure Improvement over Years")

temp = dataframe.groupby(['year'])['sufficiencyRating'].max().reset_index()
temp['sufficiencyRating'] = temp['sufficiencyRating'].astype(float)
temp.plot(y = 'sufficiencyRating', x = 'year', marker = '.', title = "Sufficiency Rating vs. Years")

temp = dataframe.groupby(['year'])['operatingRating'].max().reset_index()
temp['operatingRating'] = temp['operatingRating'].astype(float)
temp.plot(y = 'operatingRating', x = 'year', marker = '.', title = "Operating Rating vs. Years")

temp = dataframe.groupby(['year'])['structuralEvaluation'].max().reset_index()
temp['structuralEvaluation'] = temp['structuralEvaluation'].astype(float)
temp.plot(y = 'structuralEvaluation', x = 'year', marker = '.', title = "Structural Evaluation vs. Years")

