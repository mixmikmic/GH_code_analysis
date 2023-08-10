import numpy as np
import pandas as pd 

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def preprocessString(s):
    return s[:s.index('\\')]

def preprocessDataframe(original):
    original['Player'] = original['Player'].apply(preprocessString)
    original.drop('Tm', axis=1, inplace=True)
    original.dropna(inplace=True)
    return original 

def integerizeFeatures(original):
    original = pd.concat([original, pd.get_dummies(original['Pos'])], axis=1)
    original.drop('Pos', axis=1, inplace=True)
    return original

def createVectors(original):
    # Don't need the name anymore
    original.drop('Player', axis=1, inplace=True)
    return original.as_matrix(), original

# Read in data from CSV
df = pd.read_csv('NBA2016-2017Stats.csv', encoding='utf-8')

# Do some quick preprocessing
df = preprocessDataframe(df)

# Integerize the dataframe
df = integerizeFeatures(df)

# Create a list of all the players
listPlayers = df['Player'].tolist()

# Create vectors
playerVectors, df = createVectors(df)

# Create a list of all the stat categories
listStats = df.columns.tolist()

def getVector(playerName):
    indexOfPlayer = 0
    try:
        indexOfPlayer = listPlayers.index(playerName)
    except ValueError:
        print ('Player not found!')
        return
    return playerVectors[indexOfPlayer]

def showStats(vector, name):
    print ('{0}\'s stats'.format(name))
    for index,stat in enumerate(listStats):
        # Just wanna print the important stats
        if (stat in ['AST','TRB','PS/G']):
            print ('{0}: {1}'.format(stat, vector[index]))

nameOfPlayer = 'Chris Paul'
cp3Vector = getVector(nameOfPlayer)
showStats(cp3Vector, nameOfPlayer)

pca = PCA(n_components=2)
pca.fit(playerVectors)
reducedVectors = pca.transform(playerVectors)

reducedVectors[listPlayers.index('Chris Paul')]

xValues = reducedVectors[:,0]
yValues = reducedVectors[:,1]

plt.scatter(xValues, yValues, s=area, c=colors, alpha=0.5)



