from sklearn.cluster import KMeans
import numpy as np
import re
import os
import matplotlib.pyplot as plt

f = open('ALL', 'r', encoding='utf-8')

words = re.split('\W+', f.read())

f.close()

ctr = {}

for w in words:
	if w not in ctr:
		ctr[w] = 1
	else:
		ctr[w] = ctr[w] + 1

wordsAll = []

for k in iter(ctr):
	if(ctr[k] > 4 and k != "t" and k != "co" and k != "https"):
		wordsAll.append(k) 
	#f.write(ctr[k] + " - " + k)

dictTemp = {}

for w in wordsAll:
	dictTemp[w] = 0

listGeneral = []
labels = []

qwe = 1

for file in os.listdir("allTweets"):
	labels.append(file)
	print(str(qwe) + " " + file)
	qwe = qwe+1
	path = "allTweets\\" + file
	f = open(path, 'r', encoding='utf-8')
	words = re.split('\W+', f.read())
	f.close()

	dic = dict(dictTemp)

	for w in words:
		if w in dic:
			dic[w] = 1

	l = []
	l2 = []
	for k in iter(dic):
		l.append([k, dic[k]])
	l.sort()
	for i in l:
		l2.append(i[1])

	listGeneral.append(l2)

for file in os.listdir("testData"):
	labels.append(file)
	print(str(qwe) + " " + file)
	qwe = qwe+1
	path = "testData\\" + file
	f = open(path, 'r', encoding='utf-8')
	words = re.split('\W+', f.read())
	f.close()

	dic = dict(dictTemp)

	for w in words:
		if w in dic:
			dic[w] = 1

	l = []
	l2 = []
	for k in iter(dic):
		l.append([k, dic[k]])
	l.sort()
	for i in l:
		l2.append(i[1])

	listGeneral.append(l2)

kmeansList = np.asarray(listGeneral)

kmeans = KMeans(n_clusters=2, random_state=0).fit(kmeansList)

print(kmeans.labels_)
print("[1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8]")

scoresSP0 = []
scoresSP1 = []
labelsSP = []
wordScoresSPx = [0] * len(listGeneral[0])

scoresSY0 = []
scoresSY1 = []
labelsSY = []
wordScoresSYx = [0] * len(listGeneral[0])

for i in range(len(listGeneral)):
	if(kmeans.labels_[i] == 0):
		scoresSP0.append(sum(listGeneral[i] * kmeans.cluster_centers_[0]))
		scoresSP1.append(sum(listGeneral[i] * kmeans.cluster_centers_[1]))
		labelsSP.append(labels[i])
		wordScoresSPx = [x + y for x, y in zip(wordScoresSPx, listGeneral[i])]
	else:
		scoresSY0.append(sum(listGeneral[i] * kmeans.cluster_centers_[0]))
		scoresSY1.append(sum(listGeneral[i] * kmeans.cluster_centers_[1]))
		labelsSY.append(labels[i])
		wordScoresSYx = [x + y for x, y in zip(wordScoresSYx, listGeneral[i])]

plt.plot(scoresSP0, scoresSP1, 'ro')

for label, x, y in zip(labelsSP, scoresSP0, scoresSP1):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='orange', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))


plt.plot(scoresSY0, scoresSY1, 'bx')

for label, x, y in zip(labelsSY, scoresSY0, scoresSY1):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.xlabel('Spor')
plt.ylabel('Siyaset')
plt.show()

sortedDict = sorted(dictTemp)
dicSP = dict(dictTemp)
dicSY = dict(dictTemp)

wordScoresSP = []
wordScoresSY = []

for i in range(len(wordScoresSPx)):
	wordScoresSP.append([wordScoresSPx[i],sortedDict[i]])
	dicSP[sortedDict[i]] = wordScoresSPx[i]
	wordScoresSY.append([wordScoresSYx[i],sortedDict[i]])
	dicSY[sortedDict[i]] = wordScoresSYx[i]

wordScoresSP.sort()
wordScoresSY.sort()

qwe = 0
print("Populer spor kelimleri:")
for i in range(1,len(wordScoresSPx)):
	j = len(wordScoresSPx) - i
	if(dicSY[wordScoresSP[j][1]] < len(labelsSY)/2):
		print(wordScoresSP[j][1] + " - " + str(wordScoresSP[j][0]))
		qwe = qwe + 1
	if(qwe == 5):
		break

qwe = 0
print("Populer siyaset kelimeleri:")
for i in range(1,len(wordScoresSY)):
	j = len(wordScoresSYx) - i
	if(dicSP[wordScoresSY[j][1]] < len(labelsSP)/2):
		print(wordScoresSY[j][1] + " - " + str(wordScoresSY[j][0]))
		qwe = qwe + 1
	if(qwe == 5):
		break

