import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

ds=pd.read_csv('titanic_data.csv')
ds.info()

ds.head()

cols_to_drop = [
    'PassengerId',
    'Name',
    'Ticket',
    'Cabin',
    'Embarked',
]

df = ds.drop(cols_to_drop, axis=1)
df.head()

def cov_sex(s):
    if s=='male':
        return 0
    elif s=='female':
        return 1
    else:
        return s

df.Sex=df.Sex.map(cov_sex)
df.head()

data=df.dropna()
data.describe()

plt.figure()
sns.heatmap(data.corr())

in_cols=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
ou_cols=['Survived']

X=data[in_cols]
y=data[ou_cols]

data=data.reset_index(drop=True)

def div_data(xdat,fkey,fval):
    xright=pd.DataFrame([],columns=xdat.columns)
    xleft=pd.DataFrame([],columns=xdat.columns)
    
    for i in range(xdat.shape[0]):
        val=xdat[fkey].loc[i]
        if val>fval:
            xright=xright.append(xdat.loc[i])
        else:
            xleft=xleft.append(xdat.loc[i])       
    return xleft,xright

def entro(col):
    pp=col.mean()
    qq=1-pp
    entr=(-1.0 * pp * np.log2(pp))+(-1.0 * qq * np.log2(qq))
    return entr

def infogain(xdata,fkey,fval):
    left ,right=div_data(xdata,fkey,fval)
    
    if left.shape[0] == 0 or right.shape[0] == 0:
        return -10000
    
    return (2*entro(xdata.Survived) - (entro(left.Survived) + entro(right.Survived)))

for fx in X.columns:
    print (fx,)
    print (infogain(data, fx, data[fx].mean()))

class DT:
    def __init__(self, depth=0, max_depth=5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None
        
        
    def train(self,xtrain):
        print (self.depth, '-'*10)
        feat= ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        gain=[]
        for f in feat:
            gain.append(infogain(xtrain,f,xtrain[f].mean()))
            
        self.fkey=feat[np.argmax(gain)]    
        self.fval=xtrain[self.fkey].mean()
        
        
        ldata, rdata=div_data(xtrain,self.fkey,self.fval)
        ldata=ldata.reset_index(drop=True)
        rdata=rdata.reset_index(drop=True)
        
        
        if (ldata.shape[0]==0 or rdata.shape[0]==0):
            if xtrain.Survived.mean()>=0.5:
                self.target='Survived'
            else:
                self.target='Dead'
            return
        
        if self.depth>=self.max_depth:
            if xtrain.Survived.mean()>=0.5:
                self.target='Survived'
            else:
                self.target='Dead'
            return
        
        self.right=DT(depth=self.depth+1,max_depth=self.max_depth)
        self.right.train(rdata)
        
        self.left=DT(depth=self.depth+1,max_depth=self.max_depth)
        self.left.train(ldata)
        
        if xtrain.Survived.mean()>=0.5:
            self.target='Survived'
        else:
            self.target='Dead'
        return
    
    def predict(self,test):
        if test[self.fkey] >= self.fval:
            if self.right is None:
                return self.target
            return self.right.predict(test)
        else:
            if self.left is None:
                return self.target
            return self.left.predict(test)

split = int(0.8 * data.shape[0])
training_data = data[:split]
testing_data = data[split:]

dt = DT()
dt.train(training_data)

for ix in testing_data.index[:10]:
    print (dt.predict(testing_data.loc[ix]))

testing_data.head(10)





