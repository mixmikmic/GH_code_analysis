from completeRun import featureClf,cv_feature
from preamble import *
from LocalDatasets import saveDict,readDict,ScoresFromPredictions,read_features,readDict, read_duration,checkForExist
import seaborn as sns
from copy import copy
plt.rcParams['savefig.dpi'] = 200
Cat = [3,20, 21, 26, 333, 334, 335,40668, 4135, 4534, 469, 46, 50]
amountList = [0.25,0.5,0.75,1]
NonCat = [1038,1043,1046,1049,1050,1063,1067,1068,1120,1176,11,12,1459,1462,1464,1466,1467,1468,1475,1476,1478,1479,1485,1487]
second = [1489, 1491, 1492, 1493, 1494, 1497, 14, 1501, 1504, 1510, 1515, 1570, 16, 18, 22, 28, 300, 30, 32, 36, 375, 37, 39,40499,40509,40, 4134, 41, 44, 4538, 458, 53, 54]
for i in second:
    NonCat.append(i)
          
cv = 10
clfNames = [ 'RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf','GaussianNB', 'BernoulliNB','GradientBoost']
# clfNames = ['RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf','GaussianNB', 'BernoulliNB']

typ = 3
for typ in [2,3]:
    for did in NonCat:
        for amount in amountList:
            featureClf(did,cv,round(amount*(readDict(did)['NumberOfFeatures']-1)),typ)
    for did in Cat:
        for amount in amountList:
            featureClf(did,cv,round(amount*(readDict(did)['NumberOfFeatures']-1)),typ)
amountList = [0.1,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,3,4,5,6]
for amount in amountList:
    for did in NonCat:
        cv_feature(did,cv,amount)
amountList = [0.5,0.6,0.7,0.75,0.8,0.9,1,2,3,4,5,6]
# for amount in amountList:
#     for did in Cat:
#         cv_feature(did,cv,amount)

didList = Cat
scores = []
scores2 = []
amountList = [0.125,0.25,0.5,0.75,1]
func = 'cvScoreFeatures5'
func2 = 'cvScoreFeatures4'
for i,did in enumerate(didList):
    scores.append([])
    scores2.append([])
    for clfName in clfNames:
        score1 = []
        score2 = []
        for amount in amountList:
            if not checkForExist(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))) or not checkForExist(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))):
                print(func,clfName,amount,did)
            score1.append(read_features(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))))
            score2.append(read_features(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))))
        scores[i].append(score1)
        scores2[i].append(score2)
score1= []
score2 = []
score3 = []
ratio = []
for i in clfNames:
    score1.append([0 for i in range(len(amountList))])
    score2.append([0 for i in range(len(amountList)+1)])
    score3.append([0 for i in range(len(amountList)+1)])
MaxClass = []
for k,x in enumerate(scores):
    for j in range(0,len(x)):    
        for i in range(0,len(x[j])):
            score1[j][i] = score1[j][i] + (scores[k][j][i][0])/len(didList)
            score2[j][0] = score2[j][0] + (scores[k][j][i][0])/(len(didList)*len(amountList))
            score2[j][i+1] = score2[j][i+1] + (scores[k][j][i][1])/len(didList)
            score3[j][0] = score3[j][0] + (scores2[k][j][i][0])/(len(didList)*len(amountList))
            score3[j][i+1] = score3[j][i+1] + (scores2[k][j][i][1])/len(didList)

for i,x in enumerate(amountList):
    amountList[i] = x*100
amountList2 = copy(amountList)
amountList.insert(0,0)
x_axis = amountList
fig, ax = plt.subplots()
_=plt.title(' accuracy against features added, categorical datasets ' )
cl = sns.hls_palette(len(score1), l=.3, s=.8)
for i in range(0,len(clfNames)):
#     _=ax.plot(amountList2, score1[i], color = cl[i])
    _=ax.plot(x_axis, score2[i],label=clfNames[i], color = cl[i])
    _=ax.scatter(x_axis, score2[i], color = cl[i])
    _=ax.plot(x_axis, score3[i], color = cl[i],ls = ':')
_=plt.xticks(x_axis,x_axis ,rotation='vertical')
_=plt.ylabel('predictive accuracy')
_=plt.xlabel('percentage added features')
fig.set_figheight(10)
fig.set_figwidth(10)
_=ax.legend()
plt.show()
listCatA = []
listCatA2 = []
for x,i in enumerate(score2):
    listCatA.append(i[0]-i[len(i)-1])
    listCatA2.append(score3[x][0]-score3[x][len(i)-1])
listCatAR = []
listCatAR2 = []
for z,i in enumerate(score2):
    listCatAR.append(0)
    listCatAR2.append(0)
    for x,j in enumerate(i):
        if x > 0:
            listCatAR[z] = listCatAR[z] + (i[x-1]-i[x])/(x_axis[x]-x_axis[x-1])
            listCatAR2[z] = listCatAR2[z] + (score3[z][x-1]-score3[z][x])/(x_axis[x]-x_axis[x-1])
    listCatAR[z] = listCatAR[z]/len(amountList)*x_axis[x]
    listCatAR2[z] = listCatAR2[z]/len(amountList)*x_axis[x]

didList = NonCat
scores = []
scores2 = []
amountList = [0.125,0.25,0.5,0.75,1]
func = 'cvScoreFeatures5'
func2 = 'cvScoreFeatures4'
for i,did in enumerate(didList):
    scores.append([])
    scores2.append([])
    for clfName in clfNames:
        score1 = []
        score2 = []
        for amount in amountList:
            if not checkForExist(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))) or not checkForExist(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))):
                print(func,clfName,amount,did)
            score1.append(read_features(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))))
            score2.append(read_features(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))))
        scores[i].append(score1)
        scores2[i].append(score2)
score1 = []
score2 = []
score3 = []
ratio = []
for i in clfNames:
    score1.append([0 for i in range(len(amountList))])
    score2.append([0 for i in range(len(amountList)+1)])
    score3.append([0 for i in range(len(amountList)+1)])
    ratio.append([])
MaxClass = []
for k,x in enumerate(scores):
    for j in range(0,len(x)):    
        for i in range(0,len(x[j])):
            score1[j][i] = score1[j][i] + (scores[k][j][i][0])/len(didList)
            score2[j][0] = score2[j][0] + (scores[k][j][i][0])/(len(didList)*len(amountList))
            score2[j][i+1] = score2[j][i+1] + (scores[k][j][i][1])/len(didList)
            score3[j][0] = score3[j][0] + (scores2[k][j][i][0])/(len(didList)*len(amountList))
            score3[j][i+1] = score3[j][i+1] + (scores2[k][j][i][1])/len(didList)
for i,x in enumerate(amountList):
    amountList[i] = x*100
amountList2 = copy(amountList)
amountList.insert(0,0)
x_axis = amountList
fig, ax = plt.subplots()
_=plt.title(' accuracy against features added, numerical datasets, dotted line are random features,straight is duplicate' )
cl = sns.hls_palette(len(score1), l=.3, s=.8)
for i in range(0,len(clfNames)):
#     _=ax.plot(amountList2, score1[i], color = cl[i],ls = ':')
    _=ax.plot(x_axis, score2[i],label=clfNames[i], color = cl[i])
    _=ax.scatter(x_axis, score2[i], color = cl[i])
    _=ax.plot(amountList, score3[i], color = cl[i],ls = ':')
_=plt.xticks(x_axis,x_axis ,rotation='vertical')
_=plt.ylabel('predictive accuracy')
_=plt.xlabel('percentage added features')
fig.set_figheight(15)
fig.set_figwidth(15)
_=ax.legend()
plt.show()
listNumA = []
listNumA2 = []
for x,i in enumerate(score2):
    listNumA.append(i[0]-i[len(i)-1])
    listNumA2.append(score3[x][0]-score3[x][len(i)-1])
listNumAR = []
listNumAR2 = []
for z,i in enumerate(score2):
    listNumAR.append(0)
    listNumAR2.append(0)
    for x,j in enumerate(i):
        if x > 0:
            listNumAR[z] = listNumAR[z] + (i[x-1]-i[x])/(x_axis[x]-x_axis[x-1])
            listNumAR2[z] = listNumAR2[z] + (score3[z][x-1]-score3[z][x])/(x_axis[x]-x_axis[x-1])
    listNumAR[z] = listNumAR[z]/len(amountList)*x_axis[x]
    listNumAR2[z] = listNumAR2[z]/len(amountList)*x_axis[x]

glist = [listCatA,listNumA,listCatA2,listNumA2]
resultsNP = np.array(glist)
df = pd.DataFrame(resultsNP.reshape(len(glist),len(listCatA)),
                  columns=clfNames,index=['categorical datasets','numerical datasets','categorical datasets random features','numerical datasets random features'])
pd.options.display.float_format = '{:.2f}'.format
df


# glist = [listCatA,listNumA]
# resultsNP = np.array(glist)
# gf = pd.DataFrame(resultsNP.reshape(len(glist),len(list3)),
#                   columns=clfNames,index=['categorical datasets','numerical datasets'])
# gf

glist = [listCatAR,listNumAR,listCatAR2,listNumAR2]
resultsNP = np.array(glist)
df2 = pd.DataFrame(resultsNP.reshape(len(glist),len(listCatAR)),
                  columns=clfNames,index=['cat datasets averaged','num datasets averaged','cat datasets random features averaged','num datasets random features averaged'])
pd.options.display.float_format = '{:.2f}'.format
df2

didList = Cat
# clfNames = ['RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf', 'GaussianNB', 'BernoulliNB']
func = 'cvScoreFeatures5'
func2 = 'cvScoreFeatures4'
dur1 = []
dur2 = []
dur3 = []
typ = 0
fig, ax = plt.subplots()
amountList = [0.25,0.5,0.75,1] 
for i,x in enumerate(clfNames):
    dur1.append([]) 
    dur2.append([])
    dur3.append([])
    for j,x in enumerate(amountList):
        dur1[i].append(0) 
        dur2[i].append(0)
        dur3[i].append(0)
    dur2[i].append(0)
    dur3[i].append(0)
for did in didList:
    for cs,clfName in enumerate(clfNames):
        for i,amount in enumerate(amountList):
            if not checkForExist(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))) and checkForExist(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))):
                print(func,clfName,amount,did)
            dur1[cs][i] = dur1[cs][i] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ]/len(didList)
            dur2[cs][i+1] = dur2[cs][i+1] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+2]/len(didList)
            dur2[cs][0] = dur2[cs][0] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ]/(len(didList)*len(amountList))
            dur3[cs][i+1] = dur3[cs][i+1] + read_duration(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+2]/len(didList)
            dur3[cs][0] = dur3[cs][0] + read_duration(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ]/(len(didList)*len(amountList))
            dur1[cs][i] = dur1[cs][i] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+1]/len(didList)
            dur2[cs][i+1] = dur2[cs][i+1] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+3]/len(didList)
            dur2[cs][0] = dur2[cs][0] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+1]/(len(didList)*len(amountList))
            dur3[cs][i+1] = dur3[cs][i+1] + read_duration(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+3]/len(didList)
            dur3[cs][0] = dur3[cs][0] + read_duration(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+1]/(len(didList)*len(amountList))


cl = sns.hls_palette(len(dur2), l=.3, s=.8)
for i,x in enumerate(amountList):
    amountList[i] = x*100
amountList2 = copy(amountList)
amountList.insert(0,0)
x_axis = amountList
for i in range(0,len(dur2)):
#     _= ax.scatter(x_axis,dur1[i], label = clfNames[i],color = cl[i] )
    _= ax.plot(x_axis,dur2[i], color = cl[i],label = clfNames[i] )
    _= ax.plot(x_axis,dur3[i], color = cl[i],ls = ':' )
_=plt.xticks(x_axis,amountList ,rotation='vertical')
_=plt.title('classification duration against features added, categorical datasets ')
_=plt.ylabel('duration(seconds)')
if func == 'cvScoreFeatures3'or func == 'cvScoreFeatures4' or func == 'cvScoreFeatures5':
    _=plt.xlabel('percentage features added')
elif func == 'cvScoreFeatures1':
    _=plt.xlabel('features removed')
#_=plt.yscale("log", nonposy='clip')
fig.set_figheight(8)
fig.set_figwidth(8)
ax.set_yscale("log", nonposy='clip')
_=plt.legend()
_=plt.show()

func = 'cvScoreFeatures5'
didList = NonCat
dur1 = []
dur2 = []
typ = 0
fig, ax = plt.subplots()
amountList = [0.25,0.5,0.75,1] 
for i,x in enumerate(clfNames):
    dur1.append([]) 
    dur2.append([])
    for j,x in enumerate(amountList):
        dur1[i].append(0) 
        dur2[i].append(0)
    dur2[i].append(0)
for did in didList:
    for cs,clfName in enumerate(clfNames):
        for i,amount in enumerate(amountList):
            if not checkForExist(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))):
                print(func,clfName,amount,did)
            dur1[cs][i] = dur1[cs][i] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ]/len(didList)
            dur2[cs][i+1] = dur2[cs][i+1] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+2]/len(didList)
            dur2[cs][0] = dur2[cs][0] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ]/(len(didList)*len(amountList))
            dur1[cs][i] = dur1[cs][i] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+1]/len(didList)
            dur2[cs][i+1] = dur2[cs][i+1] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+3]/len(didList)
            dur2[cs][0] = dur2[cs][0] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+1]/(len(didList)*len(amountList))

cl = sns.hls_palette(len(dur2), l=.3, s=.8)
amountList = [0,0.25,0.5,0.75,1] 
for i,x in enumerate(amountList):
    amountList[i] = x*100
x_axis = amountList
for i in range(0,len(dur2)):
    _= ax.plot(x_axis,dur2[i], color = cl[i],label = clfNames[i] )
_=plt.xticks(x_axis,amountList ,rotation='vertical')
_=plt.title(' duration against features added, numerical datasets ')
_=plt.ylabel('duration(seconds)')
if func == 'cvScoreFeatures3'or func == 'cvScoreFeatures4' or func == 'cvScoreFeatures5':
    _=plt.xlabel('percentage features added')
elif func == 'cvScoreFeatures1':
    _=plt.xlabel('features removed')
fig.set_figheight(8)
fig.set_figwidth(8)
ax.set_yscale("log", nonposy='clip')
_=plt.legend()
_=plt.show()

func = 'cvFeatureSTD1'
scores = []
amountList = [0.1,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,3,4,5,6]
didList = NonCat#[54,53,39,40509,1515,1043,1570,30,28,375]#NonCat
for i,did in enumerate(didList):
    scores.append([])
    for clfName in clfNames:
        score1 = []
        for amount in amountList:
            if not checkForExist(func,clfName,did,amount):
                print(func,clfName,amount,did)
            score1.append(read_features(func,clfName,did,amount))
        scores[i].append(score1)
count = []
for amount in amountList:
    count.append(str(amount))
score1= []
score2 = []
ratio = []
for i in clfNames:
    score1.append([0 for i in range(len(amountList))])
    score2.append([0 for i in range(len(amountList)+1)])
    ratio.append([])
MaxClass = []
j = 0
for k,x in enumerate(scores):
    for j in range(0,len(x)):    
        for i in range(0,len(x[j])):
            score1[j][i] = score1[j][i] + (scores[k][j][i][0])/len(didList)
            score2[j][0] = score2[j][0] + (scores[k][j][i][0])/(len(didList)*len(amountList))
            score2[j][i+1] = score2[j][i+1] + (scores[k][j][i][1])/len(didList)
amountList2 = [0]
for i in amountList:
    amountList2.append(i)
x_axis = amountList2
fig, ax = plt.subplots()
_=plt.title(' performance to randomly std difference per feature in numerical datasets ')
cl = sns.hls_palette(len(score1), l=.3, s=.8)
for i in range(0,len(clfNames)):
#     _=ax.plot(x_axis, score1[i], color = cl[i])
#     _=ax.plot(x_axis, score1[i], color = cl[i])
    
    _=ax.plot(x_axis, score2[i],label=clfNames[i], color = cl[i])
    _=ax.scatter(x_axis, score2[i], color = cl[i])
#     _=ax.plot(amountList,approx[i],ls = ':',color = cl[i])
_=plt.ylabel('predictive accuracy')
_=plt.xlabel('maximum std difference per feature')
fig.set_figheight(10)
fig.set_figwidth(10)
_=ax.legend()
plt.show()
listNum = []
for x,i in enumerate(score2):
    listNum.append(i[0]-i[len(i)-1])
listNumR = []
for z,i in enumerate(score2):
    listNumR.append(0)
    for x,j in enumerate(i):
        if x > 0:
            listNumR[z] = listNumR[z] + (i[x-1]-i[x])/(x_axis[x]-x_axis[x-1])
    listNumR[z] = listNumR[z]/len(amountList)*x_axis[x]

func = 'cvfeatureCAT2'
scores = []
amountList = [0.5,0.6,0.7,0.75,0.8,0.9,1,2,3,4,5,6]
didList = [3,20, 21, 26, 333, 334, 335, 40668, 4135, 4534, 469, 46, 50]
for i,did in enumerate(didList):
    scores.append([])
    for clfName in clfNames:
        score1 = []
        for amount in amountList:
            if not checkForExist(func,clfName,did,amount):
                print(func,clfName,amount,did)
            score1.append(read_features(func,clfName,did,amount))
        scores[i].append(score1)
count = []
for amount in amountList:
    count.append(str(amount))
score1= []
score2 = []
ratio = []
for i in clfNames:
    score1.append([0 for i in range(len(amountList))])
    score2.append([0 for i in range(len(amountList))])
    ratio.append([])
MaxClass = []
j = 0
for k,x in enumerate(scores):
    for j in range(0,len(x)):    
        for i in range(0,len(x[j])):
            score1[j][i] = score1[j][i] + (scores[k][j][i][0])/len(didList)
            score2[j][i] = score2[j][i] + (scores[k][j][i][1])/len(didList)
flipped = []
for i in amountList:
    if i > 0.5:
        flipped.append((1-1/(i+0.5))*100)
    else:
        flipped.append(0)
x_axis = flipped
fig, ax = plt.subplots()
_=plt.title(' performance to flipping categories, categorical datasets ')
cl = sns.hls_palette(len(score1), l=.3, s=.8)
for i in range(0,len(clfNames)):
#     _=ax.plot(x_axis, score1[i], color = cl[i])
    _=ax.plot(x_axis, score2[i],label=clfNames[i], color = cl[i])
    _=ax.scatter(x_axis, score2[i], color = cl[i])
_=plt.ylabel('predictive accuracy')
_=plt.xlabel('odds of flipping category')
fig.set_figheight(10)
fig.set_figwidth(10)
_=ax.legend()
plt.show()
listCat = []
for x,i in enumerate(score2):
    listCat.append(i[0]-i[len(i)-1])
listCatR = []
for z,i in enumerate(score2):
    listCatR.append(0)
    for x,j in enumerate(i):
        if x > 0:
            listCatR[z] = listCatR[z] + (i[x-1]-i[x])/(x_axis[x]-x_axis[x-1])
    listCatR[z] = listCatR[z]/len(amountList)*x_axis[x]

glist = [listCat,listCatR,listNum,listNumR]
resultsNP = np.array(glist)
df = pd.DataFrame(resultsNP.reshape(len(glist),len(listCat)),
                  columns=clfNames,index=['categorical datasets','categorical datasets averaged','numerical datasets','numerical datasets averaged'])
df

amountList = [1.5]
didList = NonCat
func = 'cvfeatureSTD1'
for i,did in enumerate(didList):
    for clfName in clfNames:
        score1 = []
        for amount in amountList:
            if not checkForExist(func,clfName,did,amount):
                print(func,clfName,amount,did)

np.polyfit(x_axis, np.log(score2[0]), 1)

def funcs(ab,x):
    return ab[0]* np.log(x) + ab[1]

approx = []
for j,x in enumerate(clfNames):
    approx.append([])
    for i in amountList:
        approx[j].append(1+funcs(np.polyfit(x_axis, np.log(score2[j]), 1),i))

from sklearn.linear_model import Ridge
ridges = []
for j,clfName in enumerate(clfNames):
    ridges.append(Ridge().fit(np.array(amountList).reshape(-1,1),np.log(score2[j][1:])))
approx = []
for j,x in enumerate(clfNames):
    approx.append(ridges[j].predict(np.array(amountList).reshape(-1,1)))
for i,x in enumerate(approx):
    for j,x2 in enumerate(approx[i]):
        approx[i][j] = math.e**x2

fig, ax = plt.subplots()
for i,x in enumerate(score2):
    plt.plot(x_axis,x)
ax.set_yscale("log", nonposy='clip')

plt.show()

from LocalDatasets import readDict
listFeatures = []
listInstances = []
listMult = []
list1 = []
for did in NonCat:
    if readDict(did)['NumberOfFeatures'] < 11:
        list1.append(did)
    listFeatures.append(readDict(did)['NumberOfFeatures'])
    listInstances.append(readDict(did)['NumberOfInstances'])
    listMult.append(readDict(did)['NumberOfFeatures']*readDict(did)['NumberOfInstances'])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(listInstances,listFeatures)
plt.xlabel('NumberOfInstances')
plt.ylabel('NumberOfFeatures')
fig.set_figheight(10)
fig.set_figwidth(10)
plt.title('Numerical datasets size indication, many low instance datasets less than 2500 instances and less than 100 features.')
plt.show()

from LocalDatasets import readDict
listFeatures = []
listInstances = []
listMult = []
list1 = []
for did in Cat:
    if readDict(did)['NumberOfFeatures'] < 11:
        list1.append(did)
    listFeatures.append(readDict(did)['NumberOfFeatures'])
    listInstances.append(readDict(did)['NumberOfInstances'])
    listMult.append(readDict(did)['NumberOfFeatures']*readDict(did)['NumberOfInstances'])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(listInstances,listFeatures)
plt.xlabel('NumberOfInstances')
plt.ylabel('NumberOfFeatures')
fig.set_figheight(10)
fig.set_figwidth(10)
plt.title('Categorical datasets size indication, many low instance datasets')
plt.show()

didList = [11,
 1038,
 1043,
 1049,
 1050,
 37,
 1063,
 39,
 40,
 41,
 1067,
 1068,
 40496,
 53,
 54,
 1464,
 1467,
 40509,
 458,
 1510,
 1515]
clfNames = ['SVC-','GradientBoost','RandomForestClassifier', 'AdaBoost','KNeighborsClassifier']
# clfNames = ['RandomForestClassifier', 'AdaBoost','KNeighborsClassifier']
scores = []
scores2 = []
scores3 = []
amountList = [0.5,1]
func = 'cvScoreFeatures5'
func2 = 'cvOptScoreFeatures5'
func3 = 'cvScoreFeatures4'
for i,did in enumerate(didList):
    scores.append([])
    scores2.append([])
    scores3.append([])
    for clfName in clfNames:        
        score1 = []
        score2 = []
        score3 = []
        for amount in amountList:            
            if not checkForExist(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))):
                print(func2,clfName,amount,did)            
            score2.append(read_features(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))))
            if clfName == 'SVC-':
                clfName = 'SVC-rbf'
                if not checkForExist(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))):
                    print(func,clfName,amount,did)
                score1.append(read_features(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))))
                if not checkForExist(func3,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))):
                    print(func3,clfName,amount,did)
                score3.append(read_features(func3,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))))
                clfName = 'SVC-'
            else:
                score1.append(read_features(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))))
                score3.append(read_features(func3,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))))
        scores[i].append(score1)
        scores2[i].append(score2)
        scores3[i].append(score3)
score1 = []
score2 = []
score3 = []
score4 = []
ratio = []
for i in clfNames:
    score1.append([0 for i in range(len(amountList))])
    score2.append([0 for i in range(len(amountList)+1)])
    score3.append([0 for i in range(len(amountList)+1)])
    score4.append([0 for i in range(len(amountList)+1)])
    ratio.append([])
MaxClass = []
for k,x in enumerate(scores):
    for j in range(0,len(x)):    
        for i in range(0,len(x[j])):
            score1[j][i] = score1[j][i] + (scores[k][j][i][0])/len(didList)
            score2[j][0] = score2[j][0] + (scores[k][j][i][0])/(len(didList)*len(amountList))
            score2[j][i+1] = score2[j][i+1] + (scores[k][j][i][1])/len(didList)
            score3[j][0] = score3[j][0] + (scores2[k][j][i][0])/(len(didList)*len(amountList))
            score3[j][i+1] = score3[j][i+1] + (scores2[k][j][i][1])/len(didList)
            score4[j][0] = score4[j][0] + (scores3[k][j][i][0])/(len(didList)*len(amountList))
            score4[j][i+1] = score4[j][i+1] + (scores3[k][j][i][1])/len(didList)
for i,x in enumerate(amountList):
    amountList[i] = x*100
amountList2 = copy(amountList)
amountList.insert(0,0)
x_axis = amountList
fig, ax = plt.subplots()
_=plt.title('accuracy against irrelevant features added, numerical datasets, dotted line are default classifiers for duplicate features,straight is optimized for duplicate features, dashed uses random noise features' )
cl = sns.hls_palette(len(score1), l=.3, s=.8)
for i in range(0,len(clfNames)):
#     _=ax.plot(amountList2, score1[i], color = cl[i],ls = ':')
    _=ax.plot(x_axis, score2[i],ls = ':')
    _=ax.scatter(x_axis, score3[i], color = cl[i])
    _=ax.plot(amountList, score3[i], color = cl[i],label=clfNames[i])
    _=ax.plot(amountList, score4[i], color = cl[i],ls = '--')
_=plt.xticks(x_axis,x_axis ,rotation='vertical')
_=plt.ylabel('predictive accuracy')
_=plt.xlabel('percentage added features')
fig.set_figheight(15)
fig.set_figwidth(15)
_=ax.legend()
plt.show()

func = 'cvOptimizeSTD'
func2 = 'cvfeatureSTD1'
clfNames = ['SVC-','GradientBoost','RandomForestClassifier', 'AdaBoost','KNeighborsClassifier']
didList = [54,458,1043,1510,1515]
didList = [11,12,37,54,458,1038,1043,1046,1049,1050,1063,1067,1068,1120,1176,1459,1462,1464,1466,1467,1468,1510,1515]
didList = [11,12,37,54,458,1038,1043]
# didList = [1510]
scores = []
scores2 = []
amountList = [0.1,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2] #,3,4,5,6]
# didList = NonCat#[54,53,39,40509,1515,1043,1570,30,28,375]#NonCat
for i,did in enumerate(didList):
    scores.append([])
    scores2.append([])
    for clfName in clfNames:
        score1 = []
        score2 = []
        for amount in amountList:
            if not checkForExist(func,clfName,did,amount):
                print(func,clfName,amount,did)
            score1.append(read_features(func,clfName,did,amount))
            if clfName == 'SVC-':                
                clfName = 'SVC-rbf'
                if not checkForExist(func2,clfName,did,amount):
                    print(func2,clfName,amount,did)
                score2.append(read_features(func2,clfName,did,amount))
                clfName = 'SVC-' 
            else:
                if not checkForExist(func2,clfName,did,amount):
                    print(func2,clfName,amount,did)
                score2.append(read_features(func2,clfName,did,amount))
        scores[i].append(score1)
        scores2[i].append(score2)
count = []
for amount in amountList:
    count.append(str(amount))
score1= []
score2 = []
score3 = []
ratio = []
for i in clfNames:
    score1.append([0 for i in range(len(amountList))])
    score2.append([0 for i in range(len(amountList)+1)])
    score3.append([0 for i in range(len(amountList)+1)])
    ratio.append([])
MaxClass = []
j = 0
for k,x in enumerate(scores):
    for j in range(0,len(x)):    
        for i in range(0,len(x[j])):
            score1[j][i] = score1[j][i] + (scores[k][j][i][0])/len(didList)
            score2[j][0] = score2[j][0] + (scores[k][j][i][0])/(len(didList)*len(amountList))
            score2[j][i+1] = score2[j][i+1] + (scores[k][j][i][1])/len(didList)
            score3[j][0] = score3[j][0] + (scores2[k][j][i][0])/(len(didList)*len(amountList))
            score3[j][i+1] = score3[j][i+1] + (scores2[k][j][i][1])/len(didList)
amountList2 = [0]
for i in amountList:
    amountList2.append(i)
x_axis = amountList2
fig, ax = plt.subplots()
_=plt.title(' performance to randomly std difference per feature in numerical datasets. straight line is optimized, dotted line is non-optimized. ')
cl = sns.hls_palette(len(score1), l=.3, s=.8)
for i in range(0,len(clfNames)):
#     _=ax.plot(x_axis, score1[i], color = cl[i])
#     _=ax.plot(x_axis, score1[i], color = cl[i])
    _=ax.plot(x_axis, score3[i], color = cl[i],ls = ':' )
    _=ax.plot(x_axis, score2[i],label=clfNames[i], color = cl[i])
    _=ax.scatter(x_axis, score2[i], color = cl[i])
#     _=ax.plot(amountList,approx[i],ls = ':',color = cl[i])
_=plt.ylabel('predictive accuracy')
_=plt.xlabel('maximum std difference per feature')
fig.set_figheight(10)
fig.set_figwidth(10)
_=ax.legend()
plt.show()
listNum = []
for x,i in enumerate(score2):
    listNum.append(i[0]-i[len(i)-1])
listNumR = []
for z,i in enumerate(score2):
    listNumR.append(0)
    for x,j in enumerate(i):
        if x > 0:
            listNumR[z] = listNumR[z] + (i[x-1]-i[x])/(x_axis[x]-x_axis[x-1])
    listNumR[z] = listNumR[z]/len(amountList)*x_axis[x]

func = 'cvOptimizeSTD'
func2 = 'cvfeatureCAT2'
clfNames = ['SVC-','GradientBoost','RandomForestClassifier', 'AdaBoost','KNeighborsClassifier']
clfNames = ['GradientBoost','RandomForestClassifier', 'AdaBoost','KNeighborsClassifier']
scores = []
scores2 = []
amountList = [0.6,0.7,0.8,0.9,1,2] #,3,4,5,6]
didList = NonCat#[54,53,39,40509,1515,1043,1570,30,28,375]#NonCat
didList = [3,20,21,26]
for i,did in enumerate(didList):
    scores.append([])
    scores2.append([])
    for clfName in clfNames:
        score1 = []
        score2 = []
        for amount in amountList:
            if not checkForExist(func,clfName,did,amount):
                print(func,clfName,amount,did)
            score1.append(read_features(func,clfName,did,amount))
            if clfName == 'SVC-':
                clfName = 'SVC-rbf'
            if not checkForExist(func2,clfName,did,amount):
                print(func2,clfName,amount,did)
            score2.append(read_features(func2,clfName,did,amount))
            if clfName == 'SVC-rbf':
                clfName = 'SVC-'
        scores[i].append(score1)
        scores2[i].append(score2)
count = []
for amount in amountList:
    count.append(str(amount))
score1= []
score2 = []
score3 = []
ratio = []
for i in clfNames:
    score1.append([0 for i in range(len(amountList))])
    score2.append([0 for i in range(len(amountList)+1)])
    score3.append([0 for i in range(len(amountList)+1)])
    ratio.append([])
MaxClass = []
j = 0
for k,x in enumerate(scores):
    for j in range(0,len(x)):    
        for i in range(0,len(x[j])):
            score1[j][i] = score1[j][i] + (scores[k][j][i][0])/len(didList)
            score2[j][0] = score2[j][0] + (scores[k][j][i][0])/(len(didList)*len(amountList))
            score2[j][i+1] = score2[j][i+1] + (scores[k][j][i][1])/len(didList)
            score3[j][0] = score3[j][0] + (scores2[k][j][i][0])/(len(didList)*len(amountList))
            score3[j][i+1] = score3[j][i+1] + (scores2[k][j][i][1])/len(didList)
amountList2 = [0]
for i in amountList:
    amountList2.append(i)
flipped = [0]
for i in amountList:
    if i > 0.5:
        flipped.append((1-1/(i+0.5))*100)
    else:
        flipped.append(0)
x_axis = flipped
# x_axis = amountList2
fig, ax = plt.subplots()
_=plt.title('Optimized classifiers performance to flipping categories, categorical datasets,  ')
cl = sns.hls_palette(len(score1), l=.3, s=.8)
for i in range(0,len(clfNames)):
#     _=ax.plot(x_axis, score1[i], color = cl[i])
#     _=ax.plot(x_axis, score1[i], color = cl[i])
    
    _=ax.plot(x_axis, score2[i], color = cl[i],ls = ':')
    _=ax.plot(x_axis, score3[i], color = cl[i],label=clfNames[i])
    _=ax.scatter(x_axis, score2[i], color = cl[i])
#     _=ax.plot(amountList,approx[i],ls = ':',color = cl[i])
_=plt.ylabel('predictive accuracy')
_=plt.xlabel('percentage flipped categories')
fig.set_figheight(10)
fig.set_figwidth(10)
_=ax.legend()
plt.show()
listNum = []
for x,i in enumerate(score2):
    listNum.append(i[0]-i[len(i)-1])
listNumR = []
for z,i in enumerate(score2):
    listNumR.append(0)
    for x,j in enumerate(i):
        if x > 0:
            listNumR[z] = listNumR[z] + (i[x-1]-i[x])/(x_axis[x]-x_axis[x-1])
    listNumR[z] = listNumR[z]/len(amountList)*x_axis[x]

listie = [11,
 1038,
 1043,
 1049,
 1050,
 37,
 1063,
 39,
 40,
 41,
 1067,
 1068,
 40496,
 53,
 54,
 1464,
 1467,
 40509,
 458,
 1510,
 1515]
# clfNames = ['RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf', 'GaussianNB', 'BernoulliNB']
clfNames = ['SVC-','GradientBoost','RandomForestClassifier', 'AdaBoost','KNeighborsClassifier']
func = 'cvScoreFeatures5'
func2 = 'cvOptScoreFeatures5'

typ = 0
amountList = [0.5,1] 
multiplier = []
for i in clfNames:
    multiplier.append([])
for joke in listie:
    dur1 = []
    dur2 = []
    dur3 = []
    for i,x in enumerate(clfNames):
        dur1.append([]) 
        dur2.append([])
        dur3.append([])
        for j,x in enumerate(amountList):
            dur1[i].append(0) 
            dur2[i].append(0)
            dur3[i].append(0)
        dur2[i].append(0)
        dur3[i].append(0)
    didList = [joke]
    for did in didList:
        for cs,clfName in enumerate(clfNames):
            for i,amount in enumerate(amountList):
                if clfName == 'SVC-':
                    clfName = 'SVC-rbf'
                if not checkForExist(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))) and checkForExist(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))):
                    print(func,clfName,amount,did)                
                dur1[cs][i] = dur1[cs][i] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ]/len(didList)
                dur2[cs][i+1] = dur2[cs][i+1] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+2]/len(didList)
                dur2[cs][0] = dur2[cs][0] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ]/(len(didList)*len(amountList))
                dur1[cs][i] = dur1[cs][i] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+1]/len(didList)
                dur2[cs][i+1] = dur2[cs][i+1] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+3]/len(didList)
                dur2[cs][0] = dur2[cs][0] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+1]/(len(didList)*len(amountList))
                if clfName == 'SVC-rbf':
                    clfName = 'SVC-'
                dur3[cs][i+1] = dur3[cs][i+1] + read_duration(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+2]/len(didList)
                dur3[cs][0] = dur3[cs][0] + read_duration(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ]/(len(didList)*len(amountList))
                dur3[cs][i+1] = dur3[cs][i+1] + read_duration(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+3]/len(didList)
                dur3[cs][0] = dur3[cs][0] + read_duration(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+1]/(len(didList)*len(amountList))

    for i,x in enumerate(dur2):
        multiplier[i].append(np.array(dur3[i])/np.array(dur2[i]))

#print duration increase
didList = [11,
 1038,
 1043,
 1049,
 1050,
 37,
 1063,
 39,
 40,
 41,
 1067,
 1068,
 40496,
 53,
 54,
 1464,
 1467,
 40509,
 458,
 1510,
 1515]
# clfNames = ['RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf', 'GaussianNB', 'BernoulliNB']
clfNames = ['SVC-','GradientBoost','RandomForestClassifier', 'AdaBoost','KNeighborsClassifier']
func = 'cvScoreFeatures5'
func2 = 'cvOptScoreFeatures5'
dur1 = []
dur2 = []
dur3 = []
typ = 0
fig, ax = plt.subplots()
amountList = [0.5,1] 
for i,x in enumerate(clfNames):
    dur1.append([]) 
    dur2.append([])
    dur3.append([])
    for j,x in enumerate(amountList):
        dur1[i].append(0) 
        dur2[i].append(0)
        dur3[i].append(0)
    dur2[i].append(0)
    dur3[i].append(0)
for did in didList:
    for cs,clfName in enumerate(clfNames):
        for i,amount in enumerate(amountList):
            if clfName == 'SVC-':
                clfName = 'SVC-rbf'
            if not checkForExist(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))) and checkForExist(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1))):
                print(func,clfName,amount,did)                
            dur1[cs][i] = dur1[cs][i] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ]/len(didList)
            dur2[cs][i+1] = dur2[cs][i+1] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+2]/len(didList)
            dur2[cs][0] = dur2[cs][0] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ]/(len(didList)*len(amountList))
            dur1[cs][i] = dur1[cs][i] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+1]/len(didList)
            dur2[cs][i+1] = dur2[cs][i+1] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+3]/len(didList)
            dur2[cs][0] = dur2[cs][0] + read_duration(func,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+1]/(len(didList)*len(amountList))
            if clfName == 'SVC-rbf':
                clfName = 'SVC-'
            dur3[cs][i+1] = dur3[cs][i+1] + read_duration(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+2]/len(didList)
            dur3[cs][0] = dur3[cs][0] + read_duration(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ]/(len(didList)*len(amountList))
            dur3[cs][i+1] = dur3[cs][i+1] + read_duration(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+3]/len(didList)
            dur3[cs][0] = dur3[cs][0] + read_duration(func2,clfName,did,round(amount*(readDict(did)['NumberOfFeatures']-1)))[typ+1]/(len(didList)*len(amountList))
            


cl = sns.hls_palette(len(dur2), l=.3, s=.8)
for i,x in enumerate(amountList):
    amountList[i] = x*100
amountList2 = copy(amountList)
amountList.insert(0,0)
x_axis = amountList
for i in range(0,len(dur2)):
#     _= ax.scatter(x_axis,dur1[i], label = clfNames[i],color = cl[i] )
    _= ax.plot(x_axis,dur2[i], color = cl[i],ls = ':' )
    _= ax.plot(x_axis,dur3[i], color = cl[i],label = clfNames[i] )
_=plt.xticks(x_axis,amountList ,rotation='vertical')
_=plt.title('classification duration against duplicate features added, numerical datasets, straight lines are optimized, dotted are default  ')
_=plt.ylabel('duration(seconds)')
if func == 'cvScoreFeatures3'or func == 'cvScoreFeatures4' or func == 'cvScoreFeatures5':
    _=plt.xlabel('percentage features added')
elif func == 'cvScoreFeatures1':
    _=plt.xlabel('features removed')
#_=plt.yscale("log", nonposy='clip')
fig.set_figheight(12)
fig.set_figwidth(12)
ax.set_yscale("log", nonposy='clip')
_=plt.legend()
_=plt.show()

