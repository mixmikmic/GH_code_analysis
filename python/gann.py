import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Cannot disclose confidential data. will prepare a data soon from public source
data = pd.read_csv('data.csv',index_col=0,header=None,names=['price'])

#Convert index to datetime
data.index = pd.to_datetime(data.index)

#For simplificity and limited resources and time I have, I work on daily data for now. The model could be extended to work with intraday data.
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
customBd=CustomBusinessDay(calendar=USFederalHolidayCalendar())
dailyData = data.resample(customBd, how={'price': 'last'})

dta = dailyData

plt.plot(dta.index,dta.price)
plt.show()

#Check if there is na, inf or -inf in data set.
len(dta) != len(dta.dropna())

len(dta)!=len(dta.replace([np.inf, -np.inf], np.nan).dropna())

#replace all na, inf or -inf data with the data of previous date
for i in range(len(dta.price)):
    if np.isnan(dta.price[i]) or dta.price[i] == np.inf or dta.price[i] == -np.inf:
        if i>0:
            dta.price[i]=dta.price[i-1]            

gaIns = pd.DataFrame(index=dta.index)

#monthly difference of averages (doa)
meanF = 20
ma = pd.rolling_mean(dta,meanF)
doa = dta - ma

#just crossed a valley
gaIns['cVal'] = (doa.shift(1)<0) & (doa>0)
#just surmounted a peak
gaIns['sPeak'] = (doa.shift(1)>0) & (doa<0)

# 3 week relative strength index
rsiF = 15
pos = (dta - dta.shift(1))/dta
pos[pos<0] = 0
neg = (dta.shift(1)-dta)/dta
neg[neg<0] = 0
rsi = 1/(1+pd.rolling_mean(neg,rsiF)/pd.rolling_mean(pos,rsiF))

#too many sales vs purchases
gaIns['sp'] = rsi < 0.3
gaIns['ps'] = rsi > 0.7

#bull bear period
bbPeriod = 5 
localM = 3
lMin = pd.rolling_min(dta,localM)
lMax = pd.rolling_max(dta,localM)

def isBull(L):
    return all(x<y for x, y in zip(L, L[1:]))

def isBear(L):
    return all(x>y for x, y in zip(L, L[1:]))

up = pd.rolling_apply(lMin,bbPeriod,lambda x: isBull(x))==1
gaIns['up'] = up 
dw = pd.rolling_apply(lMax,bbPeriod,lambda x: isBear(x))==1
gaIns['dw'] = dw 

#bull bear period has finished
gaIns['upEnd'] = up.shift(1) & (~up)
gaIns['dwEnd'] = dw.shift(1) & (~dw)

annIns = pd.DataFrame(index=dta.index)

#monthly difference of averages (normalized with MA 30)
annIns['normDoa'] = doa/ma

#3 weeks rate of change (average with MA3)
rocF = 15
roc = (dta-dta.shift(rocF))/dta.shift(rocF)
maRocF = 3
annIns['maRoc'] = pd.rolling_mean(roc,maRocF)

#3 week relative strength index
annIns['rsi'] = rsi

#monthly standard Deviation of relative price( averaged with MA10)
stdF = 30
stdMaF = 10
rollStd = pd.rolling_std(dta,stdF)
annIns['rollStdMa'] = pd.rolling_mean(rollStd,stdMaF)

#latest 5 prices
annIns['DL0'] = dta
annIns['DL1'] = dta.shift(1)
annIns['DL2'] = dta.shift(2)
annIns['DL3'] = dta.shift(3)
annIns['DL4'] = dta.shift(4)
annIns['DL5'] = dta.shift(5)

#Neural network could have multiple outputs. Actually outputing consecutive 3 prices variaion slopes is better than 1. 
# But due to limited time and resource, here I implemente one output: relative price change.
annOut = (dta-dta.shift(1))/dta.shift(1)

#divide data into train and test dataset
# proportion of training data
p = 0.1 
#I use only 10%data to train the algorithm due to my pc cannot handle more data.
#which means I used 10% data to predict 80% data.I am not surprised the result is not very good.
#if you have faster pc, please feel free to adjust p = 0.8 as usual.
split = int(len(dta.index)*p)
n = 28 #skip some early data due to rolling calcualtion.

gasTrain = gaIns[n:split]
annsTrain = annIns[n:split]
annOutTrain = annOut[n:split]

gasTest = gaIns[split:]
annsTest = annIns[split:]
annOutTest = annOut[split:]

# Back-Propagation Neural Networks

import string
import numpy as np
np.random.seed(0)
# my sigmoid function is tanha 
def sigmoid(x):
    return np.tanh(x)
# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y*y
# 1 hidden layer artifical neural network class
class ANN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = np.ones(self.ni)
        self.ah = np.ones(self.nh)
        self.ao = np.ones(self.no)
        
        # create weights and set them to small random vaules [-0.2,0.2]
        mm = 0.2
        self.wi = np.random.random((self.ni, self.nh))*2*mm-mm
        self.wo = np.random.random((self.nh, self.no))*2*mm-mm

        # last change in weights for momentum   
        self.ci = np.zeros((self.ni, self.nh))
        self.co = np.zeros((self.nh, self.no))

    def feedforward(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = np.zeros(self.no)
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = np.zeros(self.nh)
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def pred(self, x):
        predY = []
        for j in range(x.shape[0]):
            predY.append(self.feedforward(x[j]))
        return predY
        

    def train(self, x, y, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        if x.shape[0] != y.shape[0]:
            raise ValueError('x,y shape mismatches')
            
        for i in range(iterations):
            error = 0.0
            for j in range(x.shape[0]):
                inputs = x[j]
                targets = y[j]
                self.feedforward(inputs)
                error = error + self.backPropagate(targets, N, M)
        return error

import time
import random
import math

numSol = gasTrain.shape[1]

def costf(sol):
    global annsTrain 
    global rpsTrain 
    global gasTrain
    gaTrain = gasTrain.values == True
    xTrain = annsTrain.values
    yTrain = annOutTrain.values
    for i in range(len(sol)):
        if xTrain.shape[0] != gaTrain.shape[0] or yTrain.shape[0] != gaTrain.shape[0]:
            raise ValueError('x,y shape mismatches')
        
        if sol[i] == 1:   
            xTrain = xTrain[gaTrain[:,i]]
            yTrain = yTrain[gaTrain[:,i]]
            gaTrain = gaTrain[gaTrain[:,i]]
        elif sol[i] == 0:
            xTrain = xTrain[[not(x) for x in gaTrain[:,i]]]
            yTrain = yTrain[[not(x) for x in gaTrain[:,i]]]
            gaTrain = gaTrain[[not(x) for x in gaTrain[:,i]]]
    if len(xTrain) ==0 or len(yTrain) ==0:
        error = np.inf
    else:
        numIns = xTrain.shape[1]
        numOut = yTrain.shape[1]
        nn = ANN(numIns, max(numIns,numOut)+1, numOut)

        # train it with some patterns
        error = nn.train(xTrain,yTrain)
        
    return error

def geneticoptimize(domain,costf,popsize=50,maxiter=1000,step=1,
                    mutprod=0.2,elite=0.2):
    # Mutation Operation
    def mutate(vec):
        i=random.randint(0,len(domain)-1)
        if vec[i]==domain[i][0]:
            return vec[0:i]+[vec[i]+step]+vec[i+1:] 
        elif vec[i]==domain[i][1]:
            return vec[0:i]+[vec[i]-step]+vec[i+1:]
        elif random.random()<0.5:
            return vec[0:i]+[vec[i]-step]+vec[i+1:] 
        else:
            return vec[0:i]+[vec[i]+step]+vec[i+1:]
        #return vec
    # Crossover Operation
    def crossover(r1,r2):
        i=random.randint(1,len(domain)-2)
        return r1[0:i]+r2[i:]

    # Build the initial population
    pop=[]
    for i in range(popsize):
        vec=[random.randint(domain[i][0],domain[i][1]) 
            for i in range(len(domain))]
        pop.append(vec)
    # How many winners from each generation?
    topelite=int(elite*popsize)
    scores=[(costf(v),v) for v in pop[0:topelite]]
  
    # Main loop 
    for i in range(maxiter):
        for v in pop[topelite:]:
            scores.append((costf(v),v))
        scores.sort()
        #print('bs',scores)
        ranked=[v for (s,v) in scores]
    
        # Start with the pure winners
        pop=ranked[0:topelite]
        scores = scores[0:topelite]
        # Add mutated and bred forms of the winners
        while len(pop)<popsize:
            if np.random.rand()<0.2:#mutprob:
                # Mutation
                c=random.randint(0,topelite)
                pop.append(mutate(ranked[c]))
            else:      
                # Crossover
                c1=random.randint(0,topelite)
                c2=random.randint(0,topelite)
                pop.append(crossover(ranked[c1],ranked[c2]))
        # Print current best scores
        print('bs',scores)

    return scores          
    

#main function for the GA+neural network model
domain = [(0,2)]*numSol
scores = geneticoptimize(domain,costf,10) 
#I use 10 popsize to speed up the algorithm and sacrifice accuracy of results. Please increase popsize to get more meaningful results.

#Actually the simiplified algorithm has one drawback that the best solutions may not cover all dataset.
#So it makes more sense to check other top solutions as well to make sure cover all dataset.
#One thing about the algorithm could be improved to keep a set of solutions that could completely cover all dataset.

#Predict test data for the best solution for example.
sol = scores[0][1]
    
global annsTrain 
global rpsTrain 
global gasTrain
gaTrain = gasTrain.values == True
xTrain = annsTrain.values
yTrain = annOutTrain.values

for i in range(len(sol)):
    if xTrain.shape[0] != gaTrain.shape[0] or yTrain.shape[0] != gaTrain.shape[0]:
        raise ValueError('x,y shape mismatches')
        
    if sol[i] == 1:   
        xTrain = xTrain[gaTrain[:,i]]
        yTrain = yTrain[gaTrain[:,i]]
        gaTrain = gaTrain[gaTrain[:,i]]
    elif sol[i] == 0:
        xTrain = xTrain[[not(x) for x in gaTrain[:,i]]]
        yTrain = yTrain[[not(x) for x in gaTrain[:,i]]]
        gaTrain = gaTrain[[not(x) for x in gaTrain[:,i]]]

numIns = xTrain.shape[1]
numOut = yTrain.shape[1]
nn = ANN(numIns, max(numIns,numOut)+1, numOut)

# train it with some patterns
for i in range(10):
    error = nn.train(xTrain,yTrain)
    print('error',error)   

global annsTest
global rpsTest 
global gasTest
gaTest = gasTest.values == True
xTest= annsTest.values
yTest = annOutTest.values
indexTest = gasTest.index
    
for i in range(len(sol)):
    if xTest.shape[0] != gaTest.shape[0]:
        raise ValueError('x,y shape mismatches')
        
    if sol[i] == 1:   
        xTest = xTest[gaTest[:,i]]
        yTest = yTest[gaTest[:,i]]
        indexTest = indexTest[gaTest[:,i]]
        gaTest = gaTest[gaTest[:,i]]
    elif sol[i] == 0:
        xTest = xTest[[not(x) for x in gaTest[:,i]]]
        yTest= yTest[[not(x) for x in gaTest[:,i]]]
        indexTest = indexTest[[not(x) for x in gaTest[:,i]]]
        gaTest = gaTest[[not(x) for x in gaTest[:,i]]]

# pred it 
yPred = nn.pred(xTest)

plt.plot(indexTest,yTest,indexTest,yPred)
plt.show()

