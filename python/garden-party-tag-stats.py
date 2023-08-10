from bs4 import BeautifulSoup  # For processing XMLfrom BeautifulSoup
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import itertools
from math import floor
matplotlib.style.use('ggplot')
import numpy as np

doc = open('garden-party.xml').read()
soup = BeautifulSoup(doc, 'lxml')

interps = soup.findAll('interp')

def getTags(interp): 
    descs = interp.findAll('desc')
    descList = []
    for desc in descs: 
        descList.append(desc.string)
    return descList

def getAllTags(interps):
    allTags = []
    for interp in interps: 
        tags = getTags(interp)
        for tag in tags: 
            allTags.append(tag)
    return allTags

def dedupe(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
allTags = dedupe(getAllTags(interps))
print(str(allTags))
len(allTags)

tagDict = {}
for interp in interps: 
    number = int(interp.attrs['n'])
    tags = getTags(interp)
    tagDict[number] = tags

def checkTags(tag):
    hasTags = []
    for n in tagDict: 
        if tag in tagDict[n]: 
            hasTags.append(1)
        else: 
            hasTags.append(0)
    return hasTags

hasTagMatrix = {}
for tag in allTags: 
    hasTagMatrix[tag] = checkTags(tag)
df = pd.DataFrame(hasTagMatrix)

df.head()

s = df.sum(axis='rows').sort_values(ascending=False)
mostFrequentTags = s[s>3]

mft = mostFrequentTags.plot(kind='bar', alpha=0.5, figsize=(10,5))
mft.set_xlabel('tag')
mft.set_ylabel('number of occurrences')
fig = mft.get_figure()
fig.tight_layout()
fig.savefig('images/mtf.png') # save it to a file

chunkSize=5
def chunkdf(df, chunkSize): 
    groups = df.groupby(lambda x: floor(x/chunkSize)).sum()
    return groups
groups = chunkdf(df, chunkSize)

party = [145, 150] # These are the lexia where the party occurs. Let's draw dotted lines there.
partyAdjusted = [x/chunkSize for x in party]
def plotTags(tags, thisdf=groups): 
    plot = thisdf[tags].plot(kind='area', alpha=0.5, figsize=(10,5))
    ymax = plot.get_ylim()[1]
    plot.axvspan(partyAdjusted[0], partyAdjusted[1], facecolor="0.65", alpha=0.5)
    plot.text(partyAdjusted[0]+0.2,ymax/2,'party',rotation=90)
    plot.set_xlabel('lexia number / ' + str(chunkSize))
    plot.set_ylabel('number of occurrences')
    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig('images/' + '-'.join(tags) + '.png') # save it to a file

fig = plotTags(['flora', 'sexuality'])

plotTags(['flora', 'butterflies'])

plotTags(['desire', 'eyes'])

plotTags(['flora', 'sexuality', 'death'])

plotTags(['darkness', 'light'])

plotTags(['black', 'death', 'oil'])

plotTags(['flora', 'green'])

plotTags(['green', 'light', 'black', 'darkness'])

plotTags(['hats', 'voices'])

plotTags(['sounds', 'colors', 'touch'])

plotTags(['class'])



