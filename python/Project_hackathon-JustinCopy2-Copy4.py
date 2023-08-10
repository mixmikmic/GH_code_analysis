import random
import numpy as np
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')


def points_gen(nDA,nR,nCN):
    # The function generates random values of t0 - velocity pairs
    # based on number of requested waves:
    # nDA - number of direct waves (linear moveout, label 'D')
    # nR - number of reflections (hyperbolic moveout, label 'R')
    # nCN - number of coherent noise events (linear moveout, label 'N')
    # outputs (nDA+nR+nCN) * (4) list
    # each point in the list has the following structure
    # [t0 (intercept time), velocity, flag(1=hyperbolic, 0=linear), label(see above))]
    
    
    # direct arrival
    direct = []
    n = 1
    while (n <= nDA):
        direct.append([0,random.uniform(.5,1.5),0,'D'])
        n = n+1
    
    n = 1
    reflected = []
    while (n <= nR):
        reflected.append([random.uniform(0,4),random.uniform(1.5,5),1,'R'])
        n = n+1
        
    n = 1
    noise = []
    while (n <= nCN):
        noise.append([random.uniform(-2,2),random.uniform(-3,3),0,'N'])
        n = n+1
        
    events = direct + reflected + noise
    return events

def points_plot(events):
    x = [x/1000 for x in range(0,2000,25)]
    
    fig, ax = plt.subplots()
    
    # plot waves
    for i in events:
        if i[3] == 'D':
            y = [offset/i[1] for offset in x]
            ax.plot(x,y,'r')
        if i[3] == 'N':
            ax.plot(x,[i[0]+offset/i[1] for offset in x],'b')
        if i[3] == 'R':
            ax.plot(x,[np.sqrt(i[0]**2 + offset**2 / i[1]**2) for offset in x],'g')
    
    plt.ylabel('Time, s')
    plt.xlabel('Offset, km')
    ax.set_xlim([0,2])
    ax.set_ylim([0,4])
    ax.invert_yaxis()
    ax.set_aspect(1)
    return ax



events=points_gen(3,2,3)
ax = points_plot(events)
plt.show(ax)

events2=points_gen(300,200,300)
ax2 = points_plot(events2)
plt.show(ax2)

events2

events

events[0]

events[0][0]

def makeEventsDict(events):
    eventsDict = {}
    labelsDict = {}
    # direct arrival
    labelsDict['label'] = []
    eventsDict['direct'], eventsDict['reflected'], eventsDict['coherentnoise'] = [],[],[]
    eventsDict['events'] = []
    for each in events:
#         print(each)
#         print(each[0])
#         print(each[3])
#         eventsDict['direct'].append(each[0])
#         eventsDict['reflected'].append(each[1])
#         eventsDict['coherentnoise'].append(each[2])
         eventsDict['events'].append(each[0:3])
         labelsDict['label'].append(each[3])
    return(eventsDict,labelsDict)





testEvents = makeEventsDict(events)
print(testEvents)

eventsDict = testEvents[0]
labelsDict = testEvents[1]

from sklearn import svm

from sklearn import svm
SVC = svm.SVC()
import matplotlib.pyplot as plt

# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = svm.SVC()
# clf.fit(X, y)  
X = eventsDict['events']
print("X = ",X)
y = labelsDict['label']
print("y = ",y)
clf = svm.SVC()
clf.fit(X, y) 

clf.predict([[0,1.34,0]])

# get support vectors
clf.support_vectors_

# get indices of support vectors
clf.support_ 

# get number of support vectors for each class
clf.n_support_ 

# This function combines several of the smaller lines above
# It takes a events list, turns it into two dictionaries combined, splits that into two arrays for X and Y
# trains a SVM label on them and then returns that model output details
# The model will need to be run on a input for a prediction

def comboFunctionA(events):
    testEvents = makeEventsDict(events)
    eventsDict = testEvents[0]
    labelsDict = testEvents[1]
    X = eventsDict['events']
#     print("X = ",X)
    y = labelsDict['label']
#     print("y = ",y)
    clf = svm.SVC()
    output = clf.fit(X, y)
    return(output)
    

# runs the function above and make a prediction
tempAnswer = comboFunctionA(events2)
tempAnswer

events2Model = tempAnswer

# requests a prediction from the model
clf.predict([[0,1.34,0],[0,1.4,0.2]])

# requests a prediction from the model
clf.predict([[-1.190348857444183, -0.8156719314845455, 0]])

## events2Model = tempAnswer = comboFunctionA(events2)

def testUnseenEvents__OLD(model,eventsDict):
    score = {}
    score['result'] = []
    score['test'] = []
    score['isCorrect'] = []
    x = 0
    print(eventsDict)
    for each in (eventsDict):
#         print("each",each)
#         print("each['events']" each['events'])
        prediction = model.predict(each['events'])
        print("prediction",prediction)
        print("labelsDict",eventsDict)
        print(eventsDict['label'])
        if prediction == eventsDict['label'][x]:
            isCorrect = "True"
        else:
            isCorrect = "False"
#         print(prediction == eventsDict['label']:
        score['result'].append(prediction)
        score['test'].append(each)
        score['isCorrect'].append(isCorrect)
        x = x + 1
    return(score)  


#
def testUnseenEvents(model,eventsDict):
    score = {}
    score['result'] = []
    score['test'] = []
    score['isCorrect'] = []
    x = 0
#     eventsTest_eventsDict = eventsDict[0]['events']
#     eventsTest_labelsDict = eventsDict[1]['label']
    i = 0
    while i < len(eventsDict[0]['events']):
        each = i
#         print("each",each)
        eventsTest_eventsDict = eventsDict[0]['events']
        eventsTest_labelsDict = eventsDict[1]['label']
#         print("eventsTest_eventsDict",eventsTest_eventsDict)
#         print("eventsTest_labelsDict",eventsTest_labelsDict)
#         print("eventsTest_eventsDict[i]",eventsTest_eventsDict[i])
#         print("eventsTest_labelsDict[i]",eventsTest_labelsDict[i])
        prediction = model.predict(eventsTest_eventsDict[i])
        if prediction == eventsDict[1]['label'][i]:
            isCorrect = "True"
        else:
            isCorrect = "False"
        score['result'].append(prediction)
        score['test'].append(each)
        score['isCorrect'].append(isCorrect)
        i += 1
    return(score)  

events3=points_gen(300,200,300)

events3_testEvents = makeEventsDict(events3)
events3_eventsDict = testEvents[0]
events3_labelsDict = testEvents[1]

# events3Model = comboFunctionA(events2)

answer = testUnseenEvents(events2Model,events3_testEvents)
answer

import pprint

pprint.pprint(dict(answer), width=1)

def printTable(myDict, colList=None):
   """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
   If column names (colList) aren't specified, they will show in random order.
   Author: Thierry Husson - Use it as you want but don't blame me.
   """
   if not colList: colList = list(myDict[0].keys() if myDict else [])
   myList = [colList] # 1st row = header
   for item in myDict: myList.append([str(item[col] or '') for col in colList])
   colSize = [max(map(len,col)) for col in zip(*myList)]
   formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
   myList.insert(1, ['-' * i for i in colSize]) # Seperating line
   for item in myList: print(formatStr.format(*item))

# answer

# print('events3_pred_results')
# for isCorrect, test in answer.items():
#     print('{} {}'.format(isCorrect,test))

def getTrueFalseOfPred(predictionDict):
    numberTrueFalse = {'True':0,'False':0,'PerTrue':0,'PerFalse':0,'Total':0}
    numberTrueFalse['Total'] = len(predictionDict['isCorrect'])
    i = 0
    while i < len(predictionDict['isCorrect']):
        if predictionDict['isCorrect'][i]:
            numberTrueFalse['True'] = numberTrueFalse['True'] + 1
        else:
            numberTrueFalse['False'] = numberTrueFalse['False'] + 1 
        i = i + 1
    numberTrueFalse['PerTrue'] = numberTrueFalse['True'] / numberTrueFalse['Total']
    numberTrueFalse['PerFalse'] = numberTrueFalse['False'] / numberTrueFalse['Total']
    return(numberTrueFalse)
    

getTrueFalseOfPred(answer)



