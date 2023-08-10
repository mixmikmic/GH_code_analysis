get_ipython().run_line_magic('pylab', 'inline')
import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
import sklearn
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import sqlalchemy
sns.set_style("white")

db_name = "appliedda"
hostname = "10.10.2.10"
conn = psycopg2.connect(database=db_name, host = hostname) #database connection

#load some of Avishek's function defintions to help with model comparison
def plot_precision_recall_n(y_true, y_prob, model_name):
    """
    y_true: ls 
        ls of ground truth labels
    y_prob: ls
        ls of predic proba from model
    model_name: str
        str of model name (e.g, LR_123)
    """
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax1.set_ylim(0,1.05)
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax2.set_ylim(0,1.05)
    
    name = model_name
    plt.title(name)
    plt.show()
    plt.clf()
    
def precision_at_k(y_true, y_scores,k):
    
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores ])
    return precision_score(y_true, y_pred)

select_statement = """SELECT new_id, sex, rootrace, edlevel, workexp, martlst, homeless, benefit_type,
                    has_job_q1, has_job_q2, has_job_q3, has_job_q4, wage_q1, wage_q2, wage_q3, wage_q4,
                    has_job_win1yr, lose_job_win1yr, total_wage_1yr, new_spell_win1yr, new_spell_win1yr_benefit,
                    district FROM class2.for_inference_example WHERE total_wage_1yr IS NOT NULL;"""
df = pd.read_sql( select_statement, conn )
print df.shape
df.head()

df['sex_miss'] = (df['sex'] == 0)
df['race_miss'] = (df['rootrace'] == 0)
df['ed_miss'] = (df['edlevel'] == None)
df['mar_miss'] = (df['martlst'] == 0)
df['home_miss'] = (df['homeless'] == None)
df.head()

#sex
df['male'] = (df['sex'] == 1)
df['female'] = (df['sex'] == 2)
#rootrace
df['nhwhite'] = (df['rootrace'] == 1)
df['nhblack'] = (df['rootrace'] == 2)
df['native'] = (df['rootrace'] == 3)
df['hispanic'] = (df['rootrace'] == 6)
df['asian'] = (df['rootrace'] == 7)
#edlevel
less_list = ['A', 'B', 'C', 'D', 1, 2, 3]
somehs_list = ['E', 'F', 4]
hsgrad_list = ['G', 'H', 'V', 5]
somecoll_list = ['W', 'X', 'Y', 6]
collgrad_list = ['Z', 'P', 7]
df['lessthanhs'] = (df['edlevel'].isin(less_list))
df['somehs'] = (df['edlevel'].isin(somehs_list))
df['hsgrad'] = (df['edlevel'].isin(hsgrad_list))
df['somecoll'] = (df['edlevel'].isin(somecoll_list))
df['collgrad'] = (df['edlevel'].isin(collgrad_list))
#workexp
df['noattach'] = (df['workexp'] == 0)
df['nowkexp'] = (df['workexp'] == 1)
df['prof'] = (df['workexp'] == 2)
df['othermgr'] = (df['workexp'] == 3)
df['clerical'] = (df['workexp'] == 4)
df['sales'] = (df['workexp'] == 5)
df['crafts'] = (df['workexp'] == 6)
df['oper'] = (df['workexp'] == 7)
df['service'] = (df['workexp'] == 8)
df['labor'] = (df['workexp'] == 9)
#martlst
df['nvrmar'] = (df['martlst'] == 1)
df['marwspouse'] = (df['martlst'] == 2)
df['marwospouse'] = (df['martlst'].isin([3,4,6]))
df['sepordiv'] = (df['martlst'].isin([5,7]))
df['widow'] = (df['martlst'] == 8)
#homeless
df['nothomeless'] = (df['homeless'] == 'N')
df['ishomeless'] = (df['homeless'].isin(['1','2','3','4','Y']))
#benefit_type
df['foodstamp'] = (df['benefit_type'] == 'foodstamp')
df['tanf'] = (df['benefit_type'] == 'tanf46')
df['grant'] = (df['benefit_type'] == 'grant')
#create features df

df_features = df[['male', 'female', 'nhwhite', 'nhblack', 'native', 'hispanic', 'asian', 'lessthanhs', 
                  'somehs', 'hsgrad', 'somecoll', 'collgrad', 'noattach', 'nowkexp', 'prof', 'othermgr',
                  'clerical', 'sales', 'crafts', 'oper', 'service', 'labor', 'nvrmar', 'marwspouse', 
                  'sepordiv', 'widow', 'nothomeless', 'ishomeless', 'foodstamp', 'tanf', 'grant']].copy()
# features df with qtr based job variables
df_features_wjobqtr = df[['has_job_q1', 'has_job_q2', 'has_job_q3', 'has_job_q4',
                'wage_q1', 'wage_q2', 'wage_q3', 'wage_q4']].copy()
df_features_wjobqtr.join(df_features) 
# features df with year based job variables
df_features_wjobyr = df[['has_job_win1yr', 'lose_job_win1yr', 'total_wage_1yr']].copy()
df_features_wjobyr = df_features_wjobyr.join(df_features)

df_label_returnany = df[['new_spell_win1yr']].copy()
df_label_returnsame = df[['new_spell_win1yr_benefit']].copy()

import statsmodels.api as sm
import statsmodels.formula.api as smf

# create one df and set up listings of variables for below
#drop reference categories (missing, male)
df_features_forlr = df_features_wjobyr[['female', 'nhwhite', 'nhblack', 'native', 'hispanic', 'asian', 'lessthanhs',
                                        'somehs', 'hsgrad', 'somecoll', 'collgrad', 'noattach', 'nowkexp', 'prof', 'othermgr', 
                                        'clerical', 'sales', 'crafts', 'oper', 'service', 'labor', 'nvrmar', 'marwspouse', 
                                        'sepordiv', 'widow', 'nothomeless', 'ishomeless', 'foodstamp', 'tanf', 'grant', 
                                        'has_job_win1yr', 'lose_job_win1yr', 'total_wage_1yr']].copy()
df_lrmodel = df_features_forlr.join(df_label_returnany)

#get list of features to build model statement
feat_list = list(df_features_forlr)
print feat_list
print len(feat_list)
length = len(feat_list)

#create string of features with plus signs
count = 0
feat_string = ''
for feature in feat_list:
    count += 1
    if count < (length):
        feat_string += feature 
        feat_string += ' + '
    else:
        feat_string += feature
## END FOR BLOCK

print feat_string

formula = "new_spell_win1yr ~ " + feat_string
print (formula)

#fit model - note the procedure is glm so you have to specify binomial in the family argument to get LR
model =smf.glm(formula=formula, data=df_lrmodel, family=sm.families.Binomial())
result = model.fit()
print (result.summary())

print (result.params)

df_lrtrain = df_lrmodel[:201540]
df_lrtest = df_lrmodel[201540:]
print df_lrtrain.shape
print df_lrtest.shape

#fit model - this time we're only fitting the model to the training set.
model =smf.glm(formula=formula, data=df_lrtrain, family=sm.families.Binomial())
result = model.fit()
print (result.summary())

from sklearn.metrics import confusion_matrix, classification_report

predictions = result.predict(df_lrtest)

pred_binary = (predictions > 0.5)


print confusion_matrix(df_lrtest['new_spell_win1yr'], pred_binary )
print classification_report(df_lrtest['new_spell_win1yr'], pred_binary, digits=3 )

plot_precision_recall_n(df_lrtest['new_spell_win1yr'], predictions, "Logistic Regression")

# create train/test sets
X_train, X_test, y_train, y_test = train_test_split(df_features_wjobyr, df_label_returnany, test_size = 0.2)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape
sel_features = list(X_train)

clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
       'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SGD':SGDClassifier(loss='log'),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, random_state=17, n_estimators=10),
        'NB': GaussianNB()}

sel_clfs = ['RF', 'ET', 'LR', 'SGD', 'GB', 'NB']

max_p_at_k = 0
for clfNM in sel_clfs:
    clf = clfs[clfNM]
    clf.fit( X_train, y_train )
    print clf
    y_score = clf.predict_proba(X_test)[:,1]
    predicted = np.array(y_score)
    expected = np.array(y_test)
    plot_precision_recall_n(expected,predicted, clfNM)
    p_at_1 = precision_at_k(expected,y_score, 0.05)  ## note that i changed the k value here from 0.01 to 0.05
    print('Precision at 5%: {:.2f}'.format(p_at_1))

#select only the two ML models
sel_clfs = ['RF', 'GB']

#here I've adapted the model loop from above to print feature importances instead of the precision/recall graph

for clfNM in sel_clfs:
    clf = clfs[clfNM]
    clf.fit( X_train, y_train )
    print clf
    y_score = clf.predict_proba(X_test)[:,1]
    predicted = np.array(y_score)
    expected = np.array(y_test)
    
    var_names = list(X_train) # get a list of variable names
        
    importances = clf.feature_importances_ # get the feature importances
    indices = np.argsort(importances)[::-1] # sort the list to get the highest importance first
    
    for f in range(X_train.shape[1]):
        print ("%d. feature (%s) importance = %f" % (f + 1, var_names[indices[f]], importances[indices[f]]))    
    

    p_at_1 = precision_at_k(expected,y_score, 0.05)
    print('Precision at 5%: {:.2f}'.format(p_at_1))
    print

df['cookcty'] = ((df['district'] >= 200) & (df['district'] <= 294))
df['downstate'] = ((df['district'] >= 10) & (df['district'] <= 115))

# add new cookcty to df for logistic regression
df_lrmodel['cookcty'] = df['cookcty']
df_lrtrain = df_lrmodel[:201540]
df_lrtest = df_lrmodel[201540:]
print df_lrtrain.shape
print df_lrtest.shape

formula += " + cookcty" 
model =smf.glm(formula=formula, data=df_lrtrain, family=sm.families.Binomial())
result = model.fit()
print (result.summary())

from sklearn.metrics import confusion_matrix, classification_report

predictions = result.predict(df_lrtest)

pred_binary = (predictions > 0.5)


print confusion_matrix(df_lrtest['new_spell_win1yr'], pred_binary )
print classification_report(df_lrtest['new_spell_win1yr'], pred_binary, digits=3 )

plot_precision_recall_n(df_lrtest['new_spell_win1yr'], predictions, "Logistic Regression")

df_features_wjobyr['cookcty'] = df['cookcty']
df_features_wjobyr['downstate'] =df['downstate']

X_train, X_test, y_train, y_test = train_test_split(df_features_wjobyr, df_label_returnany, test_size = 0.2)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape
sel_features = list(X_train)

sel_clfs = ['RF', 'ET', 'LR', 'SGD', 'GB', 'NB']
max_p_at_k = 0
for clfNM in sel_clfs:
    clf = clfs[clfNM]
    clf.fit( X_train, y_train )
    print clf
    y_score = clf.predict_proba(X_test)[:,1]
    predicted = np.array(y_score)
    expected = np.array(y_test)
    plot_precision_recall_n(expected,predicted, clfNM)
    p_at_1 = precision_at_k(expected,y_score, 0.05)
    print('Precision at 5%: {:.2f}'.format(p_at_1))

sel_clfs = ['RF', 'GB']
for clfNM in sel_clfs:
    clf = clfs[clfNM]
    clf.fit( X_train, y_train )
    print clf
    y_score = clf.predict_proba(X_test)[:,1]
    predicted = np.array(y_score)
    expected = np.array(y_test)
    
    var_names = list(X_train)
        
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for f in range(X_train.shape[1]):
        print ("%d. feature (%s) importance = %f" % (f + 1, var_names[indices[f]], importances[indices[f]]))    
    

    p_at_1 = precision_at_k(expected,y_score, 0.05)
    print('Precision at 5%: {:.2f}'.format(p_at_1))
    print

