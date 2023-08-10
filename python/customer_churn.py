import pandas as pd
import numpy as np

# read data
churn_df = pd.read_csv('data/churn.csv')

churn_df.head

col_names = churn_df.columns
print('Coulmn Names are : ')
print(col_names)
print('number of column are ', len(col_names))

print('Dimensions of data are : ', churn_df.shape)

churn_df.dtypes

# Check null
churn_df.isnull().sum()

# Describe Contineous data
churn_df.describe()

# Describe Cat variables

churn_df.describe(include=['object'])

#Create set of dependent and independent variables

churn_result = churn_df['Churn?']
y = churn_df['Churn?'].map({'True.':1, 'False.': 0})

#Another way of changing data
#y = np.where(churn_result == "True.", 1, 0)
y.head()

churn_df.columns

# Decide to drop columns

to_drop = ['State', 'Area Code', 'Phone', 'Churn?']
churn_feat_space = churn_df.drop(to_drop, axis =1)
churn_feat_space.describe()

churn_feat_space.describe(include=['object'])

# convert other cat var to numbers 
# conver yes and no to 1 and 0
yes_no_columns = ['Int\'l Plan', 'VMail Plan']
churn_feat_space[yes_no_columns] = (churn_feat_space[yes_no_columns] == 'yes').astype('int')

churn_feat_space[yes_no_columns].dtypes

features = churn_feat_space.columns
features

X = churn_feat_space.as_matrix().astype(np.float)

churn_feat_space.describe()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

get_ipython().magic('pinfo StandardScaler')

print("Feature space has %d observations and %d features"% (X.shape[0], X.shape[1]))

print('Unique target labels : ', np.unique(y))

# function for cross validation
from sklearn.cross_validation import KFold

# function for cross-validation
from sklearn.cross_validation import KFold
def run_cv(X,y,clf_class,**kwargs):
    # construct a k fold object
    kf = KFold(len(y),n_folds = 5, shuffle=True)
    y_pred = y.copy()
    for train_index,test_index in kf:
        X_train, X_test = X[train_index],X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        # Intialize a classifier with its arguments
        clf=clf_class(**kwargs)
        # model fit
        clf.fit(X_train,y_train)
        # model prediction
        y_pred[test_index]=clf.predict(X_test)
    return y_pred        

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn import metrics 

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

print('Support Vector Machine: ')
print("%f"% accuracy(y, run_cv(X, y, SVC)))

print("Random Forest:")
print("%f"% accuracy(y,run_cv(X,y,RandomForest)))

print("K Nearest Neighbor:")
print("%f"% accuracy(y,run_cv(X,y,KNN)))

from sklearn.metrics import confusion_matrix

# Create a confusion matrix
y = np.array(y)
class_names = np.unique(y)

confusion_matrices = [
("Support vector Machines",confusion_matrix(y,run_cv(X,y,SVC))),
("Random Forest",confusion_matrix(y,run_cv(X,y,RandomForest))),
("K nearest neighbors",confusion_matrix(y,run_cv(X,y,KNN)))]

confusion_matrices

confusion_matrices[0][1]

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

#Create plotting

def draw_confusion_matrix(confusion_matrices,class_names):
    fig = plt.figure()
    

    dd = [131, 132, 133]
    for i in range(3):
        ax =  fig.add_subplot(dd[i])
        ax.matshow(confusion_matrices[i][1])
        ax.set_title(confusion_matrices[i][0])
        ax.set_xlabel('Prediction')
        ax.set_ylabel('True')
#        plt.show

draw_confusion_matrix(confusion_matrices, class_names)


# function for cross-validation
from sklearn.cross_validation import KFold
def run_cv_prob(X, y, clf_class, **kwargs):
    # construct a k fold object
    kf = KFold(len(y),n_folds = 5, shuffle=True)
    y_prob = np.zeros([len(y), 2])

    for train_index,test_index in kf:
        X_train, X_test = X[train_index],X[test_index]
        y_train = y[train_index]
        
        # Intialize a classifier with its arguments
        clf= clf_class(**kwargs)
        # model fit
        clf.fit(X_train,y_train)
        # model prediction
        y_prob[test_index]=clf.predict_proba(X_test)

        return y_prob        

pred_prob = run_cv_prob(X, y, RandomForest, n_estimators =100)

pred_prob

pred_churn = pred_prob[:,1]
is_churn = (y == 1)

#Number of times a predicted probability is assigned to an observation
counts = pd.value_counts(pred_churn)

pred_prob

#Calculate true probabilities
true_prob = {}

for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    true_prob = pd.Series(true_prob)


counts = pd.concat([counts, true_prob], axis =1).reset_index()

counts.columns = ['pred_prob', 'count', 'true_prob']
counts

counts

colors = np.random.rand(len(counts))
plt.scatter(x=counts['pred_prob'], y= counts['true_prob'], s=counts['count'], c=colors, alpha=0.5)
plt.show()

from ggplot import *
get_ipython().magic('matplotlib inline')

baseline = np.mean(is_churn)
ggplot(counts,aes(x='pred_prob',y='true_prob',size='count')) +     geom_point(color='blue') +     xlim(-0.05,  1.05) + ylim(-0.05,1.05) +     ggtitle("Random Forest") +     xlab("Predicted probability") + ylab("Relative frequency of outcome")





