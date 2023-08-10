get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import warnings
import random
from datetime import datetime
random.seed(datetime.now())
warnings.filterwarnings('ignore')

from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import cross_validation # used to test classifier
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
from sklearn import metrics


# Generate random numbers and create a pandas dataframe

np.random.seed(3333) # To generate the same number sequence every time
x=np.random.uniform(-2, 2, (20, 2))
x=pd.DataFrame(x,columns=['X','Y'])


# First 10 elements of Y are -1 and the rest are 1

y=np.repeat([-1,1], [10, 10], axis=0)
y=pd.DataFrame(y,columns=['X'])


# Add 1 to the last 10 rows of x

x[y['X']==1]=x[y['X']==1]+1

# Checking if Classes are Linearly Separable

plt.figure(figsize=(15,8))
plt.scatter(x.loc[:9,'X'],x.loc[:9,'Y'],c='g')
plt.scatter(x.loc[10:19,'X'],x.loc[10:19,'Y'],c='r')
plt.show()

# Training a model on the data

linKernel = svm.SVC(kernel='linear', C = 1.0)
linKernel.fit(x[['X','Y']].values,y['X'])

# Generate Test data

xTest=np.random.uniform(-2, 2, (20, 2))
xTest=pd.DataFrame(xTest,columns=['X','Y'])


# First 10 elements of Y are -1 and the rest are 1

yTest=np.repeat([-1,1], [10, 10], axis=0)
yTest=pd.DataFrame(yTest,columns=['X'])

# Add 1 to the last 10 rows of xTest
xTest[yTest['X']==1]=xTest[yTest['X']==1]+1

# Predicted values
linKernel.predict(xTest[['X','Y']].values)

# Predicted values

linPredicted=linKernel.predict(xTest[['X','Y']].values)


# Confusion Matrix for performance of Classification 

matrix=confusion_matrix(yTest, linPredicted)
print(matrix)

True_Negative, False_Positive, False_Negative, True_Positive=confusion_matrix(yTest, linPredicted).ravel()
(True_Negative, False_Positive, False_Negative, True_Positive)

# Classification report for the SVM Kernel="linear"

report = classification_report(yTest, linPredicted)
print(report)

# Predictive Model using rbf Kernel

rbfKernel = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
rbfKernel.fit(x[['X','Y']].values,y['X'])

# Predicted values using RBF kernel
rbfPredicted=rbfKernel.predict(xTest[['X','Y']].values)

# Confusion Matrix for performance of Classification
matrix=confusion_matrix(yTest, rbfPredicted)
print(matrix)

True_Negative, False_Positive, False_Negative, True_Positive=confusion_matrix(yTest, rbfPredicted).ravel()
(True_Negative, False_Positive, False_Negative, True_Positive)

# Classification report for the SVM Kernel="RBF"

report = classification_report(yTest, rbfPredicted)
print(report)

# Predictive Model using Polynomial Kernel

polyKernel = svm.SVC(kernel='poly', degree=3, C=1.0)
polyKernel.fit(x[['X','Y']].values,y['X'])

# Predicted values using Polynomial kernel

polyPredicted=polyKernel.predict(xTest[['X','Y']].values)

# Confusion Matrix for performance of Classification

matrix=confusion_matrix(yTest, polyPredicted)
print(matrix)

# Classification report for the SVM Kernel="Polynomial"

report = classification_report(yTest,polyPredicted)
print(report)

# Predictive Model using GridSearch.CV()

parameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4,5,6,7,8,9,10], 'gamma': [0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5]}
svr = svm.SVC()
grid = GridSearchCV(svr, parameters)
grid.fit(x[['X','Y']].values,y['X'])

predictedGrid = grid.predict(xTest[['X','Y']].values)
grid_matrix = confusion_matrix(yTest, predictedGrid)
print(grid_matrix)

# Classification report for the GridSearch.cv()

report = classification_report(yTest,predictedGrid)
print(report)

field_names_df = pd.read_table('http://nikbearbrown.com/YouTube/MachineLearning/DATA/wpbc_data_field_names.txt',header=None)
field_names=field_names_df[0].tolist()
field_names

breast_cancer = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None,names = field_names)
breast_cancer.head()

#data formating ID is a non-informative column
breast_cancer = breast_cancer.drop("ID", 1)
breast_cancer.head()

breast_cancer.groupby('diagnosis').count()

breast_cancer.describe()

breast_cancer.groupby('diagnosis').median()

breast_cancer.groupby('diagnosis').mean()

def scaled_df(df):
    scaled = pd.DataFrame()
    for item in df:
        if item in df.select_dtypes(include=[np.float]):
            scaled[item] = ((df[item] - df[item].min()) / 
            (df[item].max() - df[item].min()))
        else: 
            scaled[item] = df[item]
    return scaled
breast_cancer_scaled = scaled_df(breast_cancer)

f, ax = plt.subplots(figsize=(11, 15))

ax.set_axis_bgcolor('#FFFFFF')
plt.title("Box Plot Breast Cancer Data Unscaled")
ax.set(xlim=(-.05, 1.05))
ax = sns.boxplot(data = breast_cancer[1:29], 
  orient = 'h', 
  palette = 'Set3')

f, ax = plt.subplots(figsize=(11, 15))

ax.set_axis_bgcolor('#FFFFFF')
plt.title("Box Plot Breast Cancer Data Scaled")
ax.set(xlim=(-.05, 1.05))
ax = sns.boxplot(data = breast_cancer_scaled[1:29], 
  orient = 'h', 
  palette = 'Set3')

predictor_names=field_names_df[0].tolist()
predictor_names=predictor_names[2:]
predictor_names

def rank_predictors(dat,l,f='diagnosis'):
    rank={}
    max_vals=dat.max()
    median_vals=dat.groupby(f).median()  # We are using the median as the mean is sensitive to outliers
    for p in l:
        score=np.abs((median_vals[p]['B']-median_vals[p]['M'])/max_vals[p])
        rank[p]=score
    return rank
cat_rank=rank_predictors(breast_cancer,predictor_names) 
cat_rank

cat_rank=sorted(cat_rank.items(), key=lambda x: x[1])
cat_rank

# Take the top predictors based on median difference
ranked_predictors=[]
for f in cat_rank[18:]:
    ranked_predictors.append(f[0])
ranked_predictors

X = breast_cancer_scaled[predictor_names]
#setting target
y = breast_cancer_scaled["diagnosis"]


#setting svm classifier
svc = svm.SVC(kernel='linear', C=1).fit(X, y)

print("KfoldCrossVal mean score using SVM is %s" %cross_val_score(svc,X,y,cv=10).mean())
#SVM metrics
sm = svc.fit(X_train, y_train)
y_pred = sm.predict(X_test)
print("Accuracy score using SVM is %s" %metrics.accuracy_score(y_test, y_pred))

