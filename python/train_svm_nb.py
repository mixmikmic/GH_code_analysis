import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

### import data

# ris contain RIS data: Genome_ID - Drug - RIS
ris = pd.read_pickle("/home/hermuba/res/data/annotated_RIS/anno_sps_df")
ris = ris.loc[ris['Species'] == 'Escherichia']
# df contain Genome_ID - cdhit(0101010)
df = pd.read_pickle("/home/hermuba/res/data/genePredicted/cdhit/ec0102_df")
# card contain Genome_ID - card_ARO(0101010)
card = pd.read_pickle('/home/hermuba/res/data/aro_pattern_df')
# cluster_detail contain cdhit - prevalance - card
cluster_detail = pd.read_pickle("/home/hermuba/res/data/genePredicted/cdhit/cluster_detail_df")

cluster_detail.columns

#those drugs have nearly 50 data, therefore their data are selected for training
train_drug = ['meropenem', 'gentamicin', 'ciprofloxacin', 'trimethoprim/sulfamethoxaole', 'ampicillin', 'cefazolin']
                                                
### Function to join X with y
# Input: X(dataframe with Genome ID, 01010); y(Genome ID, RIS), abx(drug name)
    # X can be df(all genes), acc(accessory only), card(card AROs), card AND acc, card merge with acc
# Output: df(aligned Genome_ID - X - y)

def join_df(X, y, abx):
    # subset y
    ris_need = ris.loc[ris['Antibiotic'] == abx][['Genome ID', 'Resistant Phenotype']] 
    # join X with y
    df_ris = pd.merge(X, ris_need, left_index = True, right_on = "Genome ID")
    # reset_index to prevent problems with cross validation and train/test split. Drop the old index
    df_ris = df_ris.reset_index(drop = True)
    return(df_ris)

### Feature selection with existing knowledge
# select accessory genes
acc_index = cluster_detail.loc[cluster_detail['prevalance'] < 1].index
# select gene clusters that are identified by card
card_index = cluster_detail.loc[cluster_detail['card_portion'] > 0].index
#
acc_card_intersect_index = list(set(card_index) & set(acc_index))

### adding feature: merging two X dataframe
card_and_acc_X = pd.merge(card,df[acc_index], left_index = True, right_index = True)

d = cluster_detail.loc[acc_card_intersect_index, :].to_excel("/home/hermuba/acc_card.xlsx")

from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold # this is the problem!!

    
def train_SVM(df):
    
    # split test, train
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["Genome ID", "Resistant Phenotype"], axis = 1)
        , df['Resistant Phenotype']
        , test_size=0.4 
        , random_state = 0)
    
    # choose estimator (our model)
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    
    # cross validation
    skf = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
    cv = skf.split(X_train, y_train)
   
    # tune hyperparameters
    gammas = np.logspace(-6, -1, 10)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gammas,
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    classifier = GridSearchCV(estimator=clf, cv=cv, param_grid=tuned_parameters)
    classifier.fit(X_train.values, np.asarray(y_train))
    
    v = classifier.cv_results_['mean_test_score'][classifier.best_index_]
    param = classifier.best_params_
    print(param)
    test = classifier.score(X_test, y_test)
    
    return([v,test])

# feature selection using pvalue
from scipy import stats
def p_value_list(df):
    r = df.loc[df['Resistant Phenotype'] == "Resistant"]
    not_r = df.loc[df['Resistant Phenotype'] != "Resistant"]
    
    p_value_ma = []
    for cluster_name in df.columns[:-2]:
        p_value = stats.ttest_ind(r[cluster_name], not_r[cluster_name])[1]
        p_value_ma.append(p_value)
    p = pd.Series(data = p_value_ma, index = df.columns[:-2])
    return(p.sort_values(ascending = True).index) # p value small are in the front

def p_feature_selection(complete_set, model):
    
    p = p_value_list(complete_set)
    t_matrix = []
    v_matrix = []
    feature = range(round(len(complete_set)/10),len(complete_set),round(len(complete_set)/10))
    print(feature)
    for a in feature:
        list_of_feature = list(p[0:a])
        list_of_col_name = list_of_feature + ['Resistant Phenotype', 'Genome ID']
        v, test = model(complete_set[list_of_col_name])
        t_matrix.append(test)
        v_matrix.append(v)
    p = plt.plot(feature,v_matrix,'r--',label = "val")
    p = plt.plot(feature,t_matrix,'b--', label = "test")
    p = plt.legend()
    plt.show()
    return(p)

for i in train_drug:
    print(i)
    d = join_df(df, ris, i)
    p_feature_selection(d, train_nb)
    p_feature_selection(d, train_SVM)
    run_all_condition(i)
    



x

p_feature_selection(join_df(df,ris,'meropenem'), train_SVM)

plt.plot(t)

def run_all_condition(abx):
    result = pd.DataFrame(columns = ['model','feature','val', 'test'])
    model = {'naive-bayes':train_nb,
             'SVM':train_SVM}
    for i in range(len(model)):
        print(i)
        model_name = list(model.keys())[i]
        
        m = list(model.values())[i]
        # accessory
        result.loc[result.shape[0]+1] = [model_name,'accessory']+m(join_df(df[acc_index],ris,abx))
        # accessory intersect with card
        result.loc[result.shape[0]+1] = [model_name,'acc intersect aro']+m(join_df(df[acc_card_intersect_index],ris,abx))
        # card ARO
        result.loc[result.shape[0]+1] = [model_name,'card aro']+m(join_df(card,ris,abx))
    return(result)

    

run_all_condition('meropenem')



