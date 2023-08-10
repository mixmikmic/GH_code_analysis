import statistics
import requests
import json
import pickle
import salty
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LINA
from scipy import stats
from scipy.stats import uniform as sp_rand
from scipy.stats import mode
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
import os
import sys
import pandas as pd
from collections import OrderedDict
from numpy.random import randint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from math import log
from time import sleep
get_ipython().magic('matplotlib inline')
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]   

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)   

class dev_model():
    def __init__(self, coef_data, data):
        self.Coef_data = coef_data
        self.Data = data
        
class prod_model():
    def __init__(self, coef_data, model):
        self.Coef_data = coef_data
        self.Model = model

paper_url = "http://ilthermo.boulder.nist.gov/ILT2/ilsearch?"    "cmp=&ncmp=1&year=&auth=&keyw=&prp=lcRG"
    
r = requests.get(paper_url)
header = r.json()['header']
papers = r.json()['res']
i = 1
data_url = 'http://ilthermo.boulder.nist.gov/ILT2/ilset?set={paper_id}'
for paper in papers[:1]:
    
    r = requests.get(data_url.format(paper_id=paper[0]))
    data = r.json()['data']
    with open("../salty/data/MELTING_POINT/%s.json" % i, "w") as outfile:
        json.dump(r.json(), outfile)
    #then do whatever you want to data like writing to a file
    sleep(0.5) #import step to avoid getting banned by server
    i += 1

###add JSON files to density.csv
outer_old = pd.DataFrame()
outer_new = pd.DataFrame()
number_of_files = 2266

for i in range(10):
    with open("../salty/data/DENSITY/%s.json" % str(i+1)) as json_file:
        
        #grab data, data headers (names), the salt name
        json_full = json.load(json_file)
        json_data = pd.DataFrame(json_full['data'])
        json_datanames = np.array(json_full['dhead'])
        json_data.columns =  json_datanames
        json_saltname = pd.DataFrame(json_full['components'])
        print(json_saltname.iloc[0][3])
        
        inner_old = pd.DataFrame()
        inner_new = pd.DataFrame()
        
        #loop through the columns of the data, note that some of the 
        #json files are missing pressure data. 
        for indexer in range(len(json_data.columns)):
            grab=json_data.columns[indexer]
            list = json_data[grab]
            my_list = [l[0] for l in list]
            dfmy_list = pd.DataFrame(my_list)
            dfmy_list.columns = [json_datanames[indexer][0]]
            inner_new = pd.concat([dfmy_list, inner_old], axis=1)
            inner_old = inner_new
            
        #add the name of the salt    
        inner_old['salt_name']=json_saltname.iloc[0][3]           
        
        #add to the growing dataframe
        outer_new = pd.concat([inner_old, outer_old], axis=0)
        outer_old = outer_new
print(outer_old)
# pd.DataFrame.to_csv(outer_old, path_or_buf='../salty/data/density.csv', index=False)

###a hacky hack solution to cleaning raw ILThermo data
# df = pd.read_csv("../salty/data/viscosity_full.csv")
df = pd.read_csv('../salty/data/density.csv',delimiter=',')
salts = pd.DataFrame(df["salt_name"])
salts = salts.rename(columns={"salt_name": "salts"})
###our data parsing was imperfect... some of the columns contain NaN
print(df.isnull().sum())
df = pd.concat([df["Temperature, K"], df["Pressure, kPa"],                    df["Specific density, kg/m<SUP>3</SUP>"]], axis=1)
df.dropna(inplace=True) #remove incomplete entries
df.reset_index(inplace=True, drop=True)
print(df.shape)

anions= []
cations= []
missed = 0
for i in range(df.shape[0]):
    if len(salts['salts'].iloc[i].split()) == 2:
        cations.append(salts['salts'].iloc[i].split()[0])
        anions.append(salts['salts'].iloc[i].split()[1])
    elif len(salts['salts'].iloc[i].split()) == 3:
        #two word cation
        if"tris(2-hydroxyethyl) methylammonium" in salts['salts'].iloc[i]:
            first = salts['salts'].iloc[i].split()[0]
            second = salts['salts'].iloc[i].split()[1]
            anions.append(salts['salts'].iloc[i].split()[2])
            cations.append(first + ' ' + second)
            
        #these strings have two word anions
        elif("sulfate" in salts['salts'].iloc[i] or        "phosphate" in salts['salts'].iloc[i] or        "phosphonate" in salts['salts'].iloc[i] or        "carbonate" in salts['salts'].iloc[i]):
            first = salts['salts'].iloc[i].split()[1]
            second = salts['salts'].iloc[i].split()[2]
            cations.append(salts['salts'].iloc[i].split()[0])
            anions.append(first + ' ' + second)
        elif("bis(trifluoromethylsulfonyl)imide" in salts['salts'].iloc[i]): 
            #this string contains 2 word cations
            first = salts['salts'].iloc[i].split()[0]
            second = salts['salts'].iloc[i].split()[1]
            third = salts['salts'].iloc[i].split()[2]
            cations.append(first + ' ' + second)
            anions.append(third)
        else:
            print(salts['salts'].iloc[i])
            missed += 1
    elif len(salts['salts'].iloc[i].split()) == 4:
        #this particular string block contains (1:1) at end of name
        if("1,1,2,3,3,3-hexafluoro-1-propanesulfonate" in salts['salts'].iloc[i]):
            first = salts['salts'].iloc[i].split()[0]
            second = salts['salts'].iloc[i].split()[1]
            cations.append(first + ' ' + second)
            anions.append(salts['salts'].iloc[i].split()[2])
        else:
            #and two word anion
            first = salts['salts'].iloc[i].split()[1]
            second = salts['salts'].iloc[i].split()[2]
            anions.append(first + ' ' + second)
            cations.append(salts['salts'].iloc[i].split()[0])
    elif("2-aminoethanol-2-hydroxypropanoate" in salts['salts'].iloc[i]):
        #one of the ilthermo salts is missing a space between cation/anion
        anions.append("2-hydroxypropanoate")
        cations.append("2-aminoethanol")
    elif len(salts['salts'].iloc[i].split()) == 5:
        if("bis[(trifluoromethyl)sulfonyl]imide" in salts['salts'].iloc[i]):
            anions.append("bis(trifluoromethylsulfonyl)imide")
            first = salts['salts'].iloc[i].split()[0]
            second = salts['salts'].iloc[i].split()[1]
            third = salts['salts'].iloc[i].split()[2]
            fourth = salts['salts'].iloc[i].split()[3]
            cations.append(first + ' ' + second + ' ' + third + ' ' + fourth)
        if("trifluoro(perfluoropropyl)borate" in salts['salts'].iloc[i]):
            anions.append("trifluoro(perfluoropropyl)borate")
            cations.append("N,N,N-triethyl-2-methoxyethan-1-aminium")    
    else:
        print(salts['salts'].iloc[i])
        missed += 1
anions = pd.DataFrame(anions, columns=["name-anion"])
cations = pd.DataFrame(cations, columns=["name-cation"])
salts=pd.read_csv('../salty/data/salts_with_smiles.csv',delimiter=',')
new_df = pd.concat([salts["name-cation"], salts["name-anion"], salts["Temperature, K"],                    salts["Pressure, kPa"], salts["Specific density, kg/m<SUP>3</SUP>"]],                   axis = 1)
print(missed)

cationDescriptors = salty.load_data("cationDescriptors.csv")
cationDescriptors.columns = [str(col) + '-cation' for col in cationDescriptors.columns]
anionDescriptors = salty.load_data("anionDescriptors.csv")
anionDescriptors.columns = [str(col) + '-anion' for col in anionDescriptors.columns]

new_df = pd.concat([cations, anions, df["Temperature, K"], df["Pressure, kPa"],                    df["Specific density, kg/m<SUP>3</SUP>"]], axis=1)
new_df = pd.merge(cationDescriptors, new_df, on="name-cation", how="right")
new_df = pd.merge(anionDescriptors, new_df, on="name-anion", how="right")
# new_df.dropna(inplace=True) #remove entires not in smiles database

cat_missing=[]
an_missing=[]
missing=[]
for i in range(new_df.shape[0]):
    if pd.isnull(new_df.iloc[i]).any() == True:
#         print(new_df.iloc[i], i)
        missing.append([new_df["name-cation"].iloc[i], new_df["name-anion"].iloc[i]])
        print(new_df["name-cation"].iloc[i], new_df["name-anion"].iloc[i], i)

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))

sort_and_deduplicate(missing)

trends = missing
output = []
for x in trends:
    if x not in output:
        output.append(x)
print(len(output))

from rdkit import Chem
from rdkit.Chem import Draw
m = Chem.MolFromSmiles("[P+](CCCCCC)(CCCCCC)(CCCCCC)(CCCCCCCCCCCCCC)")
a = Draw.MolToMPL(m)

pd.DataFrame.to_csv(new_df, path_or_buf='../salty/data/density_premodel.csv', index=False)

property_model = "density"
df = pd.DataFrame.from_csv('../salty/data/%s_premodel.csv' % property_model, index_col=None)
metaDf = df.select_dtypes(include=["object"])
dataDf = df.select_dtypes(include=[np.number])
property_scale = dataDf["Specific density, kg/m<SUP>3</SUP>"].apply(lambda x: log(float(x)))
cols = dataDf.columns.tolist()
instance = StandardScaler()
data = pd.DataFrame(instance.fit_transform(dataDf.iloc[:,:-1]), columns=cols[:-1])
df = pd.concat([data, property_scale, metaDf], axis=1)
mean_std_of_coeffs = pd.DataFrame([instance.mean_,instance.scale_], columns=cols[:-1])
viscosity_devmodel = dev_model(mean_std_of_coeffs, df)
pickle_out = open("../salty/data/%s_devmodel.pkl" % property_model, "wb")
pickle.dump(viscosity_devmodel, pickle_out)
pickle_out.close()

pickle_in = open("../salty/data/%s_devmodel.pkl" % property_model, "rb")
devmodel = pickle.load(pickle_in)
df = devmodel.Data
metaDf = df.select_dtypes(include=["object"])
dataDf = df.select_dtypes(include=[np.number])
X_train = dataDf.values[:,:-1]
Y_train = dataDf.values[:,-1]
#metaDf["Specific density, kg/m<SUP>3</SUP>"].str.split().apply(lambda x: log(float(x[0])))

#param_grid = {"alpha": sp_rand(0,0.1), "hidden_layer_sizes" : [randint(10)]}
# model = MLPRegressor(max_iter=10000,tol=1e-8)
param_grid = {"alpha": sp_rand(0.001,0.1)}
model = Lasso(max_iter=1e5,tol=1e-8)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=-1,                         n_iter=15)
grid_result = grid.fit(X_train, Y_train)
print(grid_result.best_estimator_)

iterations=2
averages=np.zeros(iterations)
variances=np.zeros(iterations)
test_MSE_array=[]

property_model = "density"
pickle_in = open("../salty/data/%s_devmodel.pkl" % property_model, "rb")
devmodel = pickle.load(pickle_in)
df = devmodel.Data
df = df.sample(frac=1)
# df["Viscosity, Pas"] = df["Viscosity, Pas"].str.split().apply(lambda x: log(float(x[0])))
metadf = df.select_dtypes(include=["object"])
datadf = df.select_dtypes(include=[np.number])

          
data=np.array(datadf)
n = data.shape[0]
d = data.shape[1]
d -= 1
n_train = int(n*0.8) #set fraction of data to be for training
n_test  = n - n_train
deslist=datadf.columns
score=np.zeros(len(datadf.columns))
feature_coefficients=np.zeros((len(datadf.columns),iterations))
test_MSE_array=[]
model_intercept_array=[]
for i in range(iterations):
    data = np.random.permutation(data) 
    X_train = np.zeros((n_train,d)) #prepare train/test arrays
    X_test  = np.zeros((n_test,d))
    Y_train = np.zeros((n_train))
    Y_test = np.zeros((n_test))

    ###sample from training set with replacement
    for k in range(n_train):
        x = randint(0,n_train)
        X_train[k] = data[x,:-1]
        Y_train[k] = (float(data[x,-1]))
    n = data.shape[0]
    ###sample from test set with replacement
    for k in range(n_test):
        x = randint(n_train,n)
        X_test[k] = data[x,:-1]
        Y_test[k] = (float(data[x,-1]))

    ###train the lasso model    
    model = Lasso(alpha=0.007115873059701538,tol=1e-10,max_iter=4000)
    model.fit(X_train,Y_train)
    
    ###Check what features are selected
    p=0
    avg_size=[]
    for a in range(len(data[0])-1):
        if model.coef_[a] != 0:
            score[a] = score[a] + 1
            feature_coefficients[a,i] = model.coef_[a] ###append the model coefs 
            p+=1
    avg_size.append(p)
    
    ###Calculate the test set MSE
    Y_hat = model.predict(X_test)
    n = len(Y_test)
    test_MSE = np.sum((Y_test-Y_hat)**2)**1/n
    test_MSE_array.append(test_MSE)
    
    ###Grab intercepts
    model_intercept_array.append(model.intercept_)
    
print("{}\t{}".format("average feature length:", np.average(avg_size)))
print("{}\t{}".format("average y-intercept:", "%.2f" % np.average(model_intercept_array)))
print("{}\t{}".format("average test MSE:", "%.2E" % np.average(test_MSE_array)))
print("{}\t{}".format("average MSE std dev:", "%.2E" % np.std(test_MSE_array)))
select_score=[]
select_deslist=[]
feature_coefficient_averages=[]
feature_coefficient_variance=[]
feature_coefficients_all=[]
for a in range(len(deslist)):
    if score[a] != 0:
        select_score.append(score[a])
        select_deslist.append(deslist[a])
        feature_coefficient_averages.append(np.average(feature_coefficients[a,:]))
        feature_coefficient_variance.append(np.std(feature_coefficients[a,:]))
        feature_coefficients_all.append(feature_coefficients[a,:])

#save the selected feature coeffs and their scores
df = pd.DataFrame(select_score, select_deslist)
df.to_pickle("../salty/data/bootstrap_list_scores.pkl")

#save the selected feature coefficients
df = pd.DataFrame(data=np.array(feature_coefficients_all).T, columns=select_deslist)
df = df.T.sort_values(by=1, ascending=False)
df.to_pickle("../salty/data/bootstrap_coefficients.pkl")

#save all the bootstrap data to create a box & whiskers plot
df = pd.DataFrame(data=[feature_coefficient_averages,                   feature_coefficient_variance], columns=select_deslist)
df = df.T.sort_values(by=1, ascending=False)
df.to_pickle("../salty/data/bootstrap_coefficient_estimates.pkl")

#save the coefficients sorted by their abs() values
df = pd.DataFrame(select_score, select_deslist)
df = df.sort_values(by=0, ascending=False).iloc[:]
cols = df.T.columns.tolist()
df = pd.read_pickle('../salty/data/bootstrap_coefficient_estimates.pkl')
df = df.loc[cols]
med = df.T.median()
med.sort_values()
newdf = df.T[med.index]
newdf.to_pickle('../salty/data/bootstrap_coefficient_estimates_top_sorted.pkl')

df = pd.DataFrame(select_score, select_deslist)
df.sort_values(by=0, ascending=False)

model = pd.read_pickle('../salty/data/bootstrap_coefficient_estimates_top_sorted.pkl')
model2 = model.abs()
df = model2.T.sort_values(by=0, ascending=False).iloc[:]
cols = df.T.columns.tolist()
df = pd.read_pickle('../salty/data/bootstrap_coefficients.pkl')
df = df.loc[cols]
med = df.T.median()
med.sort_values()
newdf = df.T[med.index]
newdf = newdf.replace(0, np.nan)
props = dict(boxes=tableau20[0], whiskers=tableau20[8], medians=tableau20[4],             caps=tableau20[6])
newdf.abs().plot(kind='box', figsize=(5,12), subplots=False, fontsize=18,        showmeans=True, logy=False, sharey=True, sharex=True, whis='range', showfliers=False,        color=props,  vert=False)
plt.xticks(np.arange(0,0.1,0.02))
print(df.shape)
# plt.savefig(filename='paper_images/Box_Plot_All_Salts.eps', bbox_inches='tight', format='eps',\
#            transparent=True)   

df = pd.read_pickle('../salty/data/bootstrap_coefficients.pkl')
med = df.T.median()
med.sort_values()
newdf = df.T[med.index]
df = newdf
for index, string in enumerate(newdf.columns):
    print(string)
    
    #get mean, std, N, and SEM from our sample
    samplemean=np.mean(df[string])
    print('sample mean', samplemean)
    samplestd=np.std(df[string],ddof=1)
    print('sample std', samplestd)
    sampleN=1000
    samplesem=stats.sem(df[string])
    print('sample SEM', samplesem)

    #t, the significance level of our sample mean is defined as
    #samplemean - 0 / standard error of sample mean
    #in other words, the number of standard deviations
    #the coefficient value is from 0
    #the t value by itself does not tell us very much
    t=(samplemean)/samplesem
    print('t', t)

    #the p-value tells us the propbability of achieving a value
    #at least as extreme as the one for our dataset if the null
    #hypothesis were true
    p=stats.t.sf(np.abs(t),sampleN-1)*2 #multiply by two for two-sided test
    print('p', p)

    #test rejection of the null hypothesis based on 
    #significance level of 0.05
    alpha=0.05
    if p < alpha:
        print('reject null hypothesis')
    else:
        print('null hypothesis accepted')

mse_scores=[]
for i in range(df.shape[0]):
    model = pd.read_pickle('../salty/data/bootstrap_coefficient_estimates_top_sorted.pkl')
    model2 = model.abs()
    df = model2.T.sort_values(by=0, ascending=False).iloc[:i]
    cols = df.T.columns.tolist()
    model = model[cols]
    cols = model.columns.tolist()
    cols.append("Specific density, kg/m<SUP>3</SUP>")
    
    property_model = "density"
    pickle_in = open("../salty/data/%s_devmodel.pkl" % property_model, "rb")
    devmodel = pickle.load(pickle_in)
    df = devmodel.Data
    df = df.sample(frac=1)
    metadf = df.select_dtypes(include=["object"])
    datadf = df.select_dtypes(include=[np.number])
    df = datadf.T.loc[cols]
    data=np.array(df.T)

    n = data.shape[0]
    d = data.shape[1]
    d -= 1
    n_train = 0#int(n*0.8) #set fraction of data to be for training
    n_test  = n - n_train

    X_train = np.zeros((n_train,d)) #prepare train/test arrays
    X_test  = np.zeros((n_test,d))
    Y_train = np.zeros((n_train))
    Y_test = np.zeros((n_test))
    X_train[:] = data[:n_train,:-1] #fill arrays according to train/test split
    Y_train[:] = (data[:n_train,-1].astype(float))
    X_test[:] = data[n_train:,:-1]
    Y_test[:] = (data[n_train:,-1].astype(float))
    Y_hat = np.dot(X_test, model.loc[0])+np.mean(Y_test[:] - np.dot(X_test[:], model.loc[0]))
    n = len(Y_test)
    test_MSE = np.sum((Y_test-Y_hat)**2)**1/n
    mse_scores.append(test_MSE) 

with plt.style.context('seaborn-whitegrid'):
    fig=plt.figure(figsize=(5,5), dpi=300)
    ax=fig.add_subplot(111)
    ax.plot(mse_scores)
    ax.grid(False)
# plt.xticks(np.arange(0,31,10))
# plt.yticks(np.arange(0,1.7,.4))

####Create dataset according to LASSO selected features
df = pd.read_pickle("../salty/data/bootstrap_list_scores.pkl")
df = df.sort_values(by=0, ascending=False)
avg_selected_features=20
df = df.iloc[:avg_selected_features]

#coeffs = mean_std_of_coeffs[cols]
property_model = "density"
pickle_in = open("../salty/data/%s_devmodel.pkl" % property_model, "rb")
devmodel = pickle.load(pickle_in)
rawdf = devmodel.Data
rawdf = rawdf.sample(frac=1)
metadf = rawdf.select_dtypes(include=["object"])
datadf = rawdf.select_dtypes(include=[np.number])
to_add=[]
for i in range(len(df)):
    to_add.append(df.index[i])
cols = [col for col in datadf.columns if col in to_add]
cols.append("Specific density, kg/m<SUP>3</SUP>")
df = datadf.T.loc[cols]

data=np.array(datadf)

n = data.shape[0]
d = data.shape[1]
d -= 1
n_train = int(n*0.8) #set fraction of data to be for training
n_test  = n - n_train

X_train = np.zeros((n_train,d)) #prepare train/test arrays
X_test  = np.zeros((n_test,d))
Y_train = np.zeros((n_train))
Y_test = np.zeros((n_test))
X_train[:] = data[:n_train,:-1] #fill arrays according to train/test split
Y_train[:] = (data[:n_train,-1].astype(float))
X_test[:] = data[n_train:,:-1]
Y_test[:] = (data[n_train:,-1].astype(float))

# Load the Python scripts that contain the Bayesian optimization code
get_ipython().magic('run gp.py')
# %run plotters.py
def sample_loss(params):
    return cross_val_score(SVC(C=10 ** params[0], gamma=10 ** params[1], random_state=12345),
                           X=data, y=target, scoring='roc_auc', cv=3).mean()

###Randomized Search NN Characterization
param_grid = {"activation": ["identity", "logistic", "tanh", "relu"],             "solver": ["lbfgs", "sgd", "adam"], "alpha": sp_rand(),             "learning_rate" :["constant", "invscaling", "adaptive"],             "hidden_layer_sizes": [randint(100)]}

model = MLPRegressor(max_iter=400,tol=1e-8)

grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,                          n_jobs=-1, n_iter=10)
grid_result = grid.fit(X_train, Y_train)

print(grid_result.best_estimator_)

model = MLPRegressor(activation='logistic', alpha=0.92078, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=75, learning_rate='constant',
       learning_rate_init=0.001, max_iter=1e8, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=1e-08, validation_fraction=0.1,
       verbose=False, warm_start=False)
model.fit(X_train,Y_train)

with plt.style.context('seaborn-whitegrid'):
    fig=plt.figure(figsize=(6,6), dpi=300)
    ax=fig.add_subplot(111)
    ax.plot(np.exp(Y_test),np.exp(model.predict(X_test)),            marker=".",linestyle="")

prodmodel = prod_model(coeffs, model)
pickle_out = open("../salty/data/%s_prodmodel.pkl" % property_model, "wb")
pickle.dump(prodmodel, pickle_out)
pickle_out.close()

