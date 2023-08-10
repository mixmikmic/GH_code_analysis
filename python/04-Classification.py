from htrc_features import FeatureReader

paths = {}
subjects = ['history', 'poetry', 'scifi']
for subject in subjects:
    with open(subject + '_paths.txt', 'r') as f:
        paths[subject] = [line[:len(line)-1] for line in f.readlines()]
        
history = FeatureReader(paths['history'])

poetry = FeatureReader(paths['poetry'])

scifi = FeatureReader(paths['scifi'])

print("Finished reading paths")

import numpy as np

wordDict = {}
i = 0 
volumes = []

print("Generating global dictionary")
volCount = 0
for vol in scifi.volumes():
    volumes.append(vol)
    tok_list = vol.tokenlist(pages=False)
    tokens = tok_list.index.get_level_values('token')
    if volCount == 200:  # first 200 from scifi volumes
        break
    volCount += 1 
    
    for token in tokens:
        if token not in wordDict.keys():
            wordDict[token] = i
            i += 1

for vol in poetry.volumes():
    volumes.append(vol)
    tok_list = vol.tokenlist(pages=False)
    tokens = tok_list.index.get_level_values('token')
    if volCount == 400:  # additional 200 from poetry volumes
        break
    volCount += 1 
    
    for token in tokens:
        if token not in wordDict.keys():
            wordDict[token] = i
            i += 1

print("Global dictionary generated")

print("Generating bag of words matrix")
dtm = np.zeros((volCount, len(wordDict.keys())))

for i, vol in enumerate(volumes):
    tok_list = vol.tokenlist(pages=False)
    counts = list(tok_list['count'])
    tokens = tok_list.index.get_level_values('token')
    
    for token, count in zip(tokens, counts):
        try:
            index = wordDict[token]
            dtm[i, index] = count
        except:
            pass
        
X = dtm
y = np.zeros((400))
y[200:400] = 1

print("Finished")

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn import cross_validation

tfidf = TfidfTransformer()
out = tfidf.fit_transform(X, y)

model = LinearSVC()

score = cross_validation.cross_val_score(model, X, y, cv=10)
print(np.mean(score))

model.fit(X, y)

feats = np.argsort(model.coef_[0])[:50]
top_scifi = [(list(feats).index(wordDict[w]) + 1, w) for w in wordDict.keys() if wordDict[w] in feats]
sorted(top_scifi)

feats = np.argsort(model.coef_[0])[-50:]
top_poetry = [(list(feats).index(wordDict[w]) + 1, w) for w in wordDict.keys() if wordDict[w] in feats]
sorted(top_poetry, key=lambda tup: tup[0])

vols = []
y_2 = []

wordDictGender = {}
subjects = [history, poetry, scifi]
index = 0
male = 0
female = 0

for subject in subjects:
    for vol in subject.volumes():
        if male == 10 and female == 10:
            break
        try:
            if str(vol.metadata['htrc_gender'][0]) == 'male':
                if male < 10:
                    vols.append(vol)
                    tok_list = vol.tokenlist(pages=False)
                    tokens = tok_list.index.get_level_values('token')
                    for token in tokens:
                        if token not in wordDictGender.keys():
                            wordDictGender[token] = index
                            index += 1
                            
                    y_2.append(0)
                    male += 1
                    
            elif str(vol.metadata['htrc_gender'][0]) == 'female':
                if female < 10:
                    vols.append(vol)
                    tok_list = vol.tokenlist(pages=False)
                    tokens = tok_list.index.get_level_values('token')
                    
                    for token in tokens:
                        if token not in wordDictGender.keys():
                            wordDictGender[token] = index
                            index += 1
                    
                    y_2.append(1)
                    female += 1
        except:
            pass
    if male == 10 and female == 10:
        break

import numpy as np

print("Generating bag of words matrix")
volCount = 20
dtm_gender = np.zeros((volCount, len(wordDictGender.keys())))

for i, vol in enumerate(vols):
    tok_list = vol.tokenlist(pages=False)
    counts = list(tok_list['count'])
    tokens = tok_list.index.get_level_values('token')
    
    for token, count in zip(tokens, counts):
        try:
            index = wordDictGender[token]
            dtm_gender[i, index] = count
        except:
            pass
        
X_2 = dtm_gender

print("Finished")

tfidf = TfidfTransformer()
out = tfidf.fit_transform(X_2, y_2)

model = LinearSVC()

score = cross_validation.cross_val_score(model, X_2, y_2, cv=10)
print(np.mean(score))

model.fit(X_2, y_2)

feats = np.argsort(model.coef_[0])[:50]
top_male = [(list(feats).index(wordDict[w]) + 1, w) for w in wordDict.keys() if wordDict[w] in feats]
sorted(top_male)

feats = np.argsort(model.coef_[0])[-50:]
top_female = [(list(feats).index(wordDict[w]) + 1, w) for w in wordDict.keys() if wordDict[w] in feats]
sorted(top_female, key=lambda tup: tup[0])

