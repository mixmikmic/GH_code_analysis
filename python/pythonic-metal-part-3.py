import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn

get_ipython().magic('matplotlib inline')

from utils import tokenizer
from wordcloud import WordCloud
import nltk
from nltk import FreqDist
import json

# from: http://neuro.imm.dtu.dk/wiki/LabMT 
# url = 'http://www.plosone.org/article/fetchSingleRepresentation.action?uri=info:doi/10.1371/journal.pone.0026752.s001'
# labmt = pd.read_csv(url, skiprows=2, sep='\t', index_col=0)
# labmt.to_csv("data/labmt.csv")

labmt = pd.read_csv("data/labmt.csv")
 
average = labmt.happiness_average.mean()
happiness = (labmt.set_index("word").happiness_average - average).to_dict()
 
def score_emotion(text):
    words = tokenizer.tokenize_strip_non_words(text)
    return sum([happiness.get(word.lower(), 0.0) for word in words]) / len(words)

with open("data/metalness.json", "r") as f:
    metalness = json.load(f)
    f.close()
    
def score_metal(text):
    words = tokenizer.tokenize_strip_non_words(text)
    return sum(metalness.get(word,0) for word in words) / len(words)    

lyrics = pd.read_csv("data/lyrics.csv")

lyrics.sample(10)

metallica_albums = {
    'andjusticeforall': 1988, 
    'deathmagnetic': 2008,
    'garageinc': 1998,
    'killemall': 1983,
    'load': 1996,
    'masterofpuppets': 1986,
    'metallicablackalbum': 1991, 
    'reload': 1997, 
    'ridethelightning': 1984,
}

metallica = (
    lyrics
    [lyrics.band_name == "metallica"]
    .groupby("album_name")
    .lyrics
    .apply(lambda x: " ".join(x))
    .reset_index()
)

metallica = metallica[metallica.album_name.isin(metallica_albums.keys())].copy()
metallica["year"] = metallica.album_name.apply(lambda x: metallica_albums.get(x))

metallica["metalness"] = metallica.lyrics.apply(score_metal)
metallica["emotion"] = metallica.lyrics.apply(score_emotion)

metallica.sort_values("year", inplace=True)

x = metallica.metalness.values
y = metallica.emotion.values
c = metallica.year.values
l = metallica.album_name.values

plt.figure(figsize=(10,10))
plt.plot(x,y)
plt.scatter(x,y,c=c, cmap="rainbow")


for xx,yy,ll in zip(x,y, l):
    plt.annotate(ll, xy=(xx,yy), fontsize=14)
    
plt.title("Metallica Emotional Arc", fontsize=16)
plt.xlabel("Metalness", fontsize=16)
plt.ylabel("Emotion",fontsize=16)

plt.savefig("resources/metallica_arc.png", bbox_inches='tight')

ironmaiden_albums = {
    'amatteroflifeanddeath':2006, 
    'bravenewworld': 2000,
    'danceofdeath': 2003, 
    'fearofthedark': 1992,
    'ironmaiden': 1980, 
    'killers': 1981,
    'noprayerforthedying': 1990,
    'pieceofmind': 1983,
    'powerslave': 1984,
    'seventhsonofaseventhson': 1988,
    'somewhereintime': 1985,
    'thebookofsouls': 2015,
    'thefinalfrontier': 2010,
    'thenumberofthebeast': 1982,
    'thexfactor': 1995,
    'virtualxi': 1998,
}

ironmaiden = (
    lyrics
    [lyrics.band_name == "ironmaiden"]
    .groupby("album_name")
    .lyrics
    .apply(lambda x: " ".join(x))
    .reset_index()
)

ironmaiden = ironmaiden[ironmaiden.album_name.isin(ironmaiden_albums.keys())].copy()
ironmaiden["year"] = ironmaiden.album_name.apply(lambda x: ironmaiden_albums.get(x))

ironmaiden["metalness"] = ironmaiden.lyrics.apply(score_metal)
ironmaiden["emotion"] = ironmaiden.lyrics.apply(score_emotion)

ironmaiden.sort_values("year", inplace=True)

x = ironmaiden.metalness.values
y = ironmaiden.emotion.values
c = ironmaiden.year.values
l = ironmaiden.album_name.values

plt.figure(figsize=(10,10))
plt.plot(x,y)
plt.scatter(x,y,c=c, cmap="rainbow")


for xx,yy,ll in zip(x,y, l):
    plt.annotate(ll, xy=(xx,yy), fontsize=14)
    
plt.title("Iron Maiden Emotional Arc", fontsize=16)
plt.xlabel("Metalness", fontsize=16)
plt.ylabel("Emotion",fontsize=16)

plt.savefig("resources/ironmaiden_arc.png", bbox_inches='tight')

import glob

hp = sorted(glob.glob("data/hp/*.txt"))

hps = []
for h in hp:
    with open(h, "r") as f:
        txt = f.read()

        hps.append(txt)

potter = pd.DataFrame(
    data={
        "text":hps,
        "book": [(h.split("/")[-1]).split(".")[0] for h in hp]
    }
)

potter["metalness"] = potter.text.apply(score_metal)
potter["emotion"] = potter.text.apply(score_emotion)

potter

x = potter.metalness.values
y = potter.emotion.values
l = potter.book.values

plt.figure(figsize=(10,10))
plt.plot(x,y)
plt.scatter(x,y)


for xx,yy,ll in zip(x,y, l):
    plt.annotate(ll, xy=(xx,yy), fontsize=14)
    
plt.title("Harry Potter Emotional Arc", fontsize=16)
plt.xlabel("Metalness", fontsize=16)
plt.ylabel("Emotion",fontsize=16)

plt.savefig("resources/harrypotter_arc.png", bbox_inches='tight')

h = pd.Series(happiness).rename("happyness")
m = pd.Series(metalness).rename("metalness")

mh = pd.concat([h,m], axis=1).dropna()

wds = set(mh.index.values)

ptr = FreqDist(w for w in tokenizer.tokenize_strip_non_words(" ".join(potter.text.values))
               if w in wds)
mtl = FreqDist(w for w in tokenizer.tokenize_strip_non_words(" ".join(metallica.lyrics.values))
               if w in wds)

wds = [x[0] for x in mtl.most_common(50)] + list(np.random.choice([x[0] for x in ptr.most_common(1000)], 50))

x = mh.ix[wds].metalness.values
y = mh.ix[wds].happyness.values
l = mh.ix[wds].index.values

fig, ax = plt.subplots(figsize=(10,10))
ax.scatter([],[])

for xx,yy,ll in zip(x,y,l):
    ax.annotate(ll, (xx, yy), fontsize=14)
    
plt.xlim(-1.5,4.5)
plt.ylim(-4, 3.5)
    
plt.xlabel("Metalness", fontsize=18)
plt.ylabel("Happyness", fontsize=18)
plt.title("Exploring the Metal/Happy Plane", fontsize=18)

plt.savefig("happy_metal_plane.png")

metallica.iloc[4].lyrics.index("love")

print(metallica.iloc[4].lyrics[1600:2000])

i = potter.iloc[2].text.index(" love ", i + 10)
i

print(potter.iloc[2].text[i - 200:i + 200])

potter

i = ironmaiden.iloc[8].lyrics.index("love ")
i

print(ironmaiden.iloc[8].lyrics[i - 100:i + 300])



