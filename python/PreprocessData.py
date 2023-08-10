import numpy as np
import pandas as pd
from scipy.misc import comb

def p_sml(r1,r2):
    '''
    parameters : r1,r2 are 2 array of arrays, e.g. [['a1','a2','a3'],['t1','t2']],[['a1','a4'],['t2','t5']]
                 w is an array, the weight parameter to adjust importance of each feature, shape is (n_feature,)
    output: sim is the similarity vector between r1 and r2
    '''
    k1 = np.intersect1d(r1[0],r2[0]).shape[0]/(1.0*min(len(r1[0]),len(r2[0])))
    k2 = np.intersect1d(r1[1],r2[1]).shape[0]/(1.0*min(len(r1[1]),len(r2[1])))
    sim = np.array((k1,k2))
    return sim 

def createFeature(df,features,k):
    '''
    input: df a dataframe contains k features and 1 label
           features is a list of k names of features to be used
           flist is a list of k names of features to be generated
    output: save pairwise features as a .csv
    '''
    X = df.ix[:,features].values
    y = df['label'].values
    n = X.shape[0]
    all_feature = []
    for i in range(n):
        for j in range(n):
            p = p_sml(X[i,],X[j,])
            is_same = int(y[i]==y[j])
            all_feature.append([(i,j),p[0],p[1],is_same])
    new_df = pd.DataFrame(all_feature)
    new_df.columns = ['pair','n_coauthor','onegram_journal','is_same']
    fpath = '../data/feature'+str(k)+'_ziwei.csv'
    new_df.to_csv(fpath)
    
    print('features sucessfully saved!')

def cleanData(i):
    fpath = '../data/text'+str(i)+'.csv'
    df = pd.read_csv(fpath)
    df.journalTitle = df.journalTitle.fillna('')
    df.coauthor = df.coauthor.fillna('')
    df.journalTitle = [x.split("|") for x in df.journalTitle.tolist()]
    df.coauthor = [x.split("|") for x in df.coauthor.tolist()]
    
    # use partial features to test the algorithm
    features = ['coauthor','journalTitle']
    label = 'authorNum'
    my_df = df[features]
    my_df = my_df.assign(label=df[label].values)
    
    return my_df

for i in range(1,15):
    df = cleanData(i) 
    createFeature(df,['coauthor','journalTitle'],i)

def combineFeature(i):
    f1 = '../data/feature'+str(i)+'_ziwei.csv'
    f2 = '../data/feature'+str(i)+'_shuyi.csv'
    f3 = '../data/feature'+str(i)+'_bo.csv'
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df3 = pd.read_csv(f3)
    df = pd.concat([df1[['n_coauthor','onegram_journal','is_same']],df2[['tfidf simlarity','edit distance','edit distance similarity']],df3[['bigram','trigram']]],1)
    df['edit distance'] = df['edit distance'].values/(df['edit distance'].max()*1.0)
    df['bigram'] = df['bigram'].values/(df['bigram'].max()*1.0)
    df['trigram'] = df['trigram'].values/(df['trigram'].max()*1.0)
    df.columns = ['coauthor','journal','is_same','tfidf_sml','edit_dist','edit_dist_sml','bigram','trigram']
    save_path = '../data/feature'+str(i)+'.csv'
    df.to_csv(save_path)
    print("Successfully combined and saved features!")

for i in range(1,15):
    combineFeature(i)



