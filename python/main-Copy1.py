import pandas_ml as pdml
import operator
import time
import matplotlib 
import matplotlib.pyplot as plt 
import sklearn
from sklearn import cross_validation,metrics,datasets, preprocessing, linear_model,svm,neighbors,grid_search,dummy
import pandas as pd 
import seaborn as sns 
import xgboost as xgb 
import numpy as np
import scipy.io
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV
get_ipython().magic('matplotlib inline')

#set seed for all randomized algos 
seed = 42 
print("preambles loaded.")

#the usual utils 


    


def load_dataset(path="/home/faker/下載/data.csv"):
    import pandas as pd
    data = pd.read_csv(path)
    print("data loaded.")
    return data

def seperator(message):
    print()
    print("-"*80)
    print(message)
    print("-"*80)
    print()
    
def make_kaggle_submit_csv(predictions): 
    return predictions

def logging(message="\n"):
    current_time = time.ctime()
    path = "/home/faker/下載/logging.txt"
    with open(path,"a") as f: 
        f.write("-"*80 + "\n")
        f.write(current_time+"\n")
        f.write(message)
        f.write("\n")
        
#plotting utils 
def plot_roc_curve(false_positive_rates,true_positive_rates,auc_score):
    plt.plot([0,1],[0,1],"k--")
    plt.plot(false_positive_rates,true_positive_rates,"r-",label = "ROC curve (auc: %0.3f)"%(auc_score))
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("false positive rate (1 - specificity)")
    plt.ylabel("true polsitive rate(sensitivity)")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.show()
    
def plot_validation_curve(train_scores,validation_scores,ticks,semilogx = False):
    if semilogx: 
        plt.semilogx(ticks,train_scores.mean(axis=1),"b--",label = "training")
        plt.fill_between(ticks,train_scores.mean(axis=1)-train_scores.std(axis=1),
                        train_scores.mean(axis=1)+train_scores.std(axis=1),alpha=0.2,color="b")
        plt.semilogx(ticks,validation_scores.mean(axis=1),"r--",label="validation")
        plt.fill_between(ticks,validation_scores.mean(axis=1)-validation_scores.std(axis=1),
                        validation_scores.mean(axis=1)+validation_scores.std(axis=1),alpha=0.2,color="r")
    else: 
        plt.plot(ticks,train_scores,"b--",label = "training")
#         plt.fill_between(ticks,train_scores.mean(axis=1-train_scores.std(axis=1),
#                         train_scores.mean(axis=1)+train_scores.std(axis=1),alpha=0.2,color="b")
        plt.plot(ticks,validation_scores,"r--",label="validation")
#         plt.fill_between(ticks,validation_scores.mean(axis=1)-validation_scores.std(axis=1),
#                         validation_scores.mean(axis=1)+validation_scores.std(axis=1),alpha=0.2,color="r")
        
    plt.legend(loc="best")
    plt.show()

def get_numpy_arrays(data_raw):
    x = mat["X"]
    y = mat["Y"] #encoded as 1,2 
    y = y.reshape((y.shape[0],))
    #change encoding of 1,2 to 0,1 
    y[y==1] = 0 
    y[y==2] = 1
    # print(y)
    return(x,y)

def get_pandas_df(numpy_x,numpy_y,verbose=False):
    feature_names = [str(i) for i in range(numpy_x.shape[1])]
    #convert to pandas df 
    data_x = pd.DataFrame(numpy_x,columns=feature_names)
    data_y = pd.Series(numpy_y,name="y")
    if verbose:
        print("convertd to pandas df!! \n")
        seperator("data_x:")
        print(data_x.info())
        seperator("top 5 rows of data_x:")
        print(data_x.head(5))
        seperator("data_y:")
        print(data_y.value_counts())
        seperator("percentage of classes:")
        print(data_y.value_counts(normalize=True))
        
    return(data_x,data_y)

def combine_train_x_y_df(data_x,data_y):
    data = pd.concat([data_x,data_y],axis=1)
    return data

def visualize_pair_wise(data,feautures=None,cols_index_range = None,target="SHEDDING_SC1"):
    
    #see all pairwise relationship between features 
    if cols_index_range:
        f = data.columns[cols_index_range]
    else:
        f = feautures
    assert len(f) <= 10, "common man, too many features to fit in a plot!"
    
    g = sns.PairGrid(data,vars=f,hue=target)
    g.map_diag(sns.kdeplot)
    g.map_upper(sns.regplot)
    g.map_lower(sns.residplot)
    plt.legend()
    plt.show()




def visualize_distro(data,features=None,cols_index_range = None,target="SHEDDING_SC1"): 
    #sumary stats
    if cols_index_range: 
        tiny_x= data.iloc[:,cols_index_range]
    else:
        tiny_x= data.loc[:,features]
    for f in tiny_x: 
        seperator("featuer:%s"%f)
        #vanilla dist plot 
        sns.distplot(tiny_x[f])
        plt.show()
        #I want a fucking distplot wiht the two classes 
        g = sns.FacetGrid(data=data,hue=target)
        g.map(sns.distplot,f)
        plt.legend()
        plt.show()
        print(tiny_x[f].describe())

def visualize_scatter(data,feature1=None,feature2=None,feature_idx1=None,feature_idx2=None,target ="SHEDDING_SC1" ):
    if feature_idx1 and feature_idx2:
        f1 = data.columns[feature_idx1]
        f2 = data.columns[feature_idx2]
    else:
        f1 = feature1
        f2 = feature2
    seperator("feature:%s Vs. feature:%s"%(f1,f2))
    sns.lmplot(data=data,x=f1,y=f2,order=1,hue=target,markers=["o","x"])
    plt.show()
    g = sns.FacetGrid(data=data,hue=target)
    g.map(sns.residplot,f1,f2)

def visaulize_muliple_curves():
    pass 


######################33
#from koby uncleaned 

def make_kaggle_submit(prediction,file_name):
    idx = pd.Series(data["shot_id"][data["y"].isnull()].values.astype(dtype =int))
    predict = pd.Series(prediction)
    submit = pd.concat([idx,predict],axis=1,ignore_index=True,join="inner")
    submit.columns = ["shot_id","shot_made_flag"]
    submit.to_csv(base_path+file_name,index=False)
    return submit
    
def score_prediction(prediction,label):
    from sklearn import metrics 
    log_loss = metrics.log_loss(label,prediction,normalize=True)
    auc_score = metrics.roc_auc_score(label,prediction)
    accuracy_score = metrics.accuracy_score(label,prediction)
    print("log loss:%5.10f, auc:%5.f, accuracy:%.f"%(log_loss,auc_score,accuracy_score))
    
    
#begin with categorical features
def visualize_categoricals(train,cols_index_range = range(10),target = "SHEDDING_SC1"):

    for f in train.columns[cols_index_range]:
        if train[f].dtype == "object":
            seperator(message=f)
            print(train[f].value_counts())
            print(train[f].value_counts(normalize = True,sort=True,ascending=False))
            sns.countplot(data=train,x=f,hue=target)
            plt.show()
            print(train[[f,target]].groupby(f).mean())

# # the numerical features 
# for f in train:
#     if train[f].dtype != "object":#numerical 
#         seperator(message=f)
#         sns.distplot(a=train[f],hist=True,kde=True,rug=False)
#         plt.show()
#         sns.boxplot(x="y",y=f,data=train)
#         plt.show()
#         print(train[f].describe())



mat = load_dataset()

data = mat 

data

train_df = pdml.ModelFrame(data, target='SHEDDING_SC1')
train_df



train_df=train_df.drop('CEL', 1)

df=train_df

#最強的df
df

train_df, test_df = train_df.cross_validation.train_test_split()
#now the train_df  , test_df  已經分開了

xgc = train_df.xgboost.XGBClassifier(objective="binary:logistic")
xgc

train_df.fit(xgc)

train_df.cross_validation.cross_val_score(xgc, cv=3, scoring='log_loss')

train_df.fit(xgc, eval_metric='mlogloss')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15, 90))
train_df.xgboost.plot_importance(ax=ax)

# Predicting...
predicted = test_df.predict(xgc)
predicted

predicted.value_counts()

test_df.metrics.confusion_matrix()





















train_df.columns[12]


features = list(train_df.columns[13:])

y_train = train_df.SHEDDING_SC1
y_train

x_train = train_df[features]

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

ceate_feature_map(features)

xgb_params = {"objective": "reg:linear", "eta": 0.01, "max_depth": 8, "seed": 42, "silent": 1}
num_rounds = 1000

dtrain = xgb.DMatrix(x_train, label=y_train)
gbdt = xgb.train(xgb_params, dtrain, num_rounds)


importance = gbdt.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1),reverse=True)

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')

df

#current x_train 2371 rows × 22277 columns (2371, 22277) , y_train
x_train

#train_x, test_x, train_y, test_y = cross_validation.train_test_split(data_x.values,data_y.values,test_size=47,random_state=seed,stratify=data_y.values)
#X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5)
#X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train,test_size=0.2,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, random_state=0)


x_train.shape

y_train.size



train_df.target

train_df = train_df.drop('Unnamed: 0', 1)
#remove useless feature

train_df

kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=rng)

xgc = train_df.xgboost.XGBClassifier

xgc





















visualize_categoricals(train=data,cols_index_range=range(10))

visualize_distro(data,features=None,cols_index_range = range(100,110,1),target="SHEDDING_SC1")

visualize_pair_wise(data,cols_index_range=range(100,105,1))

visualize_scatter(data,feature_idx1=100,feature_idx2=101)

data["TIMEHOURS"].value_counts()

for f in data.columns:
    print(f)

data_before = data.iloc[data["TIMEHOURS"]<=0,:]
print(data_before.info())
print(data_before.head())

data["SUBJECTID"].value_counts()



