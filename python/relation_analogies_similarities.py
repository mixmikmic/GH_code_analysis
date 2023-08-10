import pandas as pd
import numpy as np

df = pd.DataFrame(pd.read_csv("./data/minimal.50d.3f.csv")).set_index("0")

words = df.index

lens = (df**2).sum(axis=1)
dfn = df.div(np.sqrt(lens), axis='index')

def find_most_similar(word, n=5):
    return dfn.dot(dfn.loc[word]).sort_values(ascending=False).head(n)

def riddle(x, y, a, n = 5):
    return dfn.dot(dfn.loc[a] + dfn.loc[y] - dfn.loc[x]).sort_values(ascending=False).head(n)

find_most_similar("probably", 10)

find_most_similar("blue", 10)

find_most_similar("dance", 10)

find_most_similar("europe", 10)

find_most_similar("vinci", 10)

riddle("man", "king", "woman")

find_most_similar("king")

riddle("warsaw", "poland", "moscow")

riddle("good", "bad", "up")

riddle("france", "paris", "poland")

riddle("france", "paris", "cambodia")

riddle("city", "country", "hamburg")

riddle("hope", "disappointment", "peace")

riddle("country", "language", "britain")

dfn.dot(dfn.loc["germany"] - dfn.loc["country"] + dfn.loc["language"]).sort_values(ascending=False).head(5)

riddle("science", "einstein", "painting")

riddle("astronomy", "copernicus", "philosophy")

dfn.dot(dfn.loc["merkel"] - dfn.loc["germany"] + dfn.loc["america"]).sort_values(ascending=False).head(5)



