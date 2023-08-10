from bs4 import BeautifulSoup
btree = BeautifulSoup(open("./Melbourne_bike_share.xml"),"lxml-xml") 

print(btree.prettify())

featuretags = btree.find_all("featurename")
featuretags

for feature in featuretags:
    print (feature.string)

featurenames = [feature.string for feature in btree.find_all("featurename")]

featurenames

nbbikes = [feature.string for feature in btree.find_all("nbbikes")]
nbbikes

NBEmptydoc = [feature.string for feature in btree.find_all("nbemptydoc")]
NBEmptydoc

TerminalNames = [feature.string for feature in btree.find_all("terminalname")]
TerminalNames

UploadDate = [feature.string for feature in btree.find_all("uploaddate")]
UploadDate

ids = [feature.string for feature in btree.find_all("id")]
ids

lattitudes = [coord["latitude"] for coord in btree.find_all("coordinates")]
lattitudes

longitudes = [coord["longitude"] for coord in btree.find_all("coordinates")]
longitudes

import pandas as pd 
dataDict = {}
dataDict['Featurename'] = featurenames
dataDict['TerminalName'] = TerminalNames
dataDict['NBBikes'] = nbbikes
dataDict['NBEmptydoc'] = NBEmptydoc
dataDict['UploadDate'] = UploadDate
dataDict['lat'] = lattitudes
dataDict['lon'] = longitudes
df = pd.DataFrame(dataDict, index = ids)
df.index.name = 'ID'
df.head()



