get_ipython().magic('pylab inline')
import pandas as pd
import sklearn
from rdkit import rdBase
import matplotlib as mpl
import numpy as np
import seaborn as sns
print('RDKit version: ',rdBase.rdkitVersion)
print('Pandas version:', pd.__version__)
print('Scikit-Learn version:', sklearn.__version__)
print('Numpy version:', np.__version__)
print('MatplotLib version:', mpl.__version__)
print('Seaborn version',sns.__version__)

from rdkit.Chem.Draw import IPythonConsole

trainSet = pd.read_csv('train.smi',
                       encoding='iso-8859-1',
                       na_values=['?'],
                       delim_whitespace=True,
                       names=['Number','ID','Name','logS','Detail','Detail2','Smiles'])[['Number','ID','Name','logS','Smiles']]

trainSet.head()

from rdkit import Chem
from rdkit.Chem import PandasTools

trainSet.dropna(axis='index',inplace=True)

PandasTools.AddMoleculeColumnToFrame(frame=trainSet, smilesCol='Smiles', molCol='Molecule')

problematicSmiles = trainSet.ix[trainSet['Molecule'].map(lambda x: x is None)]
problematicSmiles

# "trainSet['Molecule'].map(lambda x: x is not None)" is generating a pandas Series with Booleans
#  that can be directly used as a row filter
trainSet = trainSet.ix[trainSet['Molecule'].map(lambda x: x is not None)]



trainSet.describe()

sns.violinplot(trainSet['logS'],cut=0,scale="count", inner="quartile")

from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
descriptors = list(np.array(Descriptors._descList)[:,0])
print(np.array(Descriptors._descList)[:,0])

calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)

def computeDescriptors(mol, calculator):
    res = np.array(calculator.CalcDescriptors(mol))
    if not np.all(np.isfinite(res)):
        return None  #make is easier to identify problematic molecules (.e.g infity descriptor values) later 
    return res



#compute the descriptors
trainSet['Descriptors'] = trainSet['Molecule'].map(lambda x: computeDescriptors(x,calculator))
#and remove rows with non-finite descriptor values (seems to be only 1 by comparing the counts)
trainSet = trainSet.ix[trainSet['Descriptors'].map(lambda x: x is not None)]
trainSet.describe()

trainSet.head()

figure,(ax1,ax2,ax3) = pyplot.subplots(1,3)
figure.set_size_inches(15,15)
#simple linear regression
from sklearn import linear_model

simple_linearreg = linear_model.LinearRegression(normalize=True)
simple_linearreg.fit(trainSet['Descriptors'].tolist(),trainSet['logS'])
simple_prediction = simple_linearreg.predict(trainSet['Descriptors'].tolist())
ax1.scatter(trainSet['logS'],simple_prediction)
ax1.set_aspect('equal')
ax1.set_title('Simple linear regression')
ax1.set_xlabel('Measured logS')
ax1.set_ylabel('Predicted logS')
ax1.set_xlim(-12,2)
ax1.set_ylim(-12,2)

#simple decision tree regression
from sklearn import tree
simple_tree = tree.DecisionTreeRegressor()
simple_tree.fit(trainSet['Descriptors'].tolist(),trainSet['logS'])
simple_treeprediction = simple_tree.predict(trainSet['Descriptors'].tolist())
ax2.scatter(trainSet['logS'],simple_treeprediction)
ax2.set_aspect('equal')
ax2.set_title('Default Decision Tree')
ax2.set_xlabel('Measured logS')
ax2.set_ylabel('Predicted logS')
ax2.set_xlim(-12,2)
ax2.set_ylim(-12,2)

#custom decision tree regression
from sklearn import tree
custom_tree = tree.DecisionTreeRegressor(max_depth=10, min_samples_split = 50)
custom_tree.fit(trainSet['Descriptors'].tolist(),trainSet['logS'])
custom_treeprediction = custom_tree.predict(trainSet['Descriptors'].tolist())
ax3.scatter(trainSet['logS'],custom_treeprediction)
ax3.set_aspect('equal')
ax3.set_title('Custom Decision Tree (regularized)')
ax3.set_xlabel('Measured logS')
ax3.set_ylabel('Predicted logS')
ax3.set_xlim(-12,2)
ax3.set_ylim(-12,2)

from sklearn.metrics import mean_squared_error
print('Coefficient of determination R^2 (LR): ',simple_linearreg.score(trainSet['Descriptors'].tolist(),trainSet['logS']))
print('MSE (LR): ',(mean_squared_error(trainSet['logS'],simple_prediction)))
print('Coefficient of determination R^2 (Default Tree): ',(simple_tree.score(trainSet['Descriptors'].tolist(),trainSet['logS'])))
print('MSE (Default Tree): ',(mean_squared_error(trainSet['logS'],simple_treeprediction)))
print('Coefficient of determination R^2 (Custom Tree): ',(custom_tree.score(trainSet['Descriptors'].tolist(),trainSet['logS'])))
print('MSE (Custom Tree): ',(mean_squared_error(trainSet['logS'],custom_treeprediction)))
#print zip(simple_prediction,custom_prediction)

def createImportancePlot(splt,desc,importances,caption):
    labels = []
    weights = []
    threshold = sort([abs(w) for w in importances])[max(-11,-(len(importances)))]
    for d in zip(desc,importances):
        if abs(d[1]) >= threshold:
            labels.append(d[0])
            weights.append(d[1])
    xlocations = np.array(range(len(labels)))+0.5
    width = 0.8
    splt.bar(xlocations, weights, width=width)
    splt.set_xticks([r+1 for r in range(len(labels))])
    splt.set_xticklabels(labels,rotation=30,)
    splt.set_xlim(0, xlocations[-1]+width*2)
    splt.set_title(caption)
    splt.get_xaxis().tick_bottom()
    splt.get_yaxis().tick_left()

figure,(plt1,plt2,plt3) = pyplot.subplots(3,1)
figure.set_size_inches(15,10)
figure.subplots_adjust(hspace=0.5)
imp = simple_linearreg.coef_
createImportancePlot(plt1,descriptors,imp,"Most important descriptors in linear model (coefficients)")
imp2 = simple_tree.feature_importances_ #Gini importances
createImportancePlot(plt2,descriptors,imp2,"Most important descriptors in simple tree model (Gini importances)")
imp3 = custom_tree.feature_importances_ #Gini importances
createImportancePlot(plt3,descriptors,imp3,"Most important descriptors in custom tree model (Gini importances)")

figure,(ax1,ax2,ax3) = pyplot.subplots(1,3)
figure.set_size_inches(15,5)

#inspect logP contribution
from scipy.stats.stats import pearsonr
ind = descriptors.index('MolLogP')
logP = [d[ind] for d in trainSet['Descriptors']]
ax1.scatter(logP,trainSet['logS'])
ax1.set_title('rdkit:mollogP vs logS')
ax1.set_xlabel('rdkit:mollogP')
ax1.set_ylabel('Measured logS')

ax2.scatter([d[descriptors.index('LabuteASA')] for d in trainSet['Descriptors']],trainSet['logS'])
ax2.set_title('rdkit:LabuteASA vs logS')
ax2.set_xlabel('rdkit:LabuteASA')
ax2.set_ylabel('Measured logS')

ax3.scatter([d[descriptors.index('HeavyAtomCount')] for d in trainSet['Descriptors']],trainSet['logS'])
ax3.set_title('rdkit:HeavyAtomCount vs logS')
ax3.set_xlabel('rdkit:HeavyAtomCount')
ax3.set_ylabel('Measured logS')

print('Pearson Correlation (MolLogP): ',pearsonr(logP,trainSet['logS'])[0])
print('Pearson Correlation (LabuteASA): ',pearsonr([d[descriptors.index('LabuteASA')] for d in trainSet['Descriptors']],trainSet['logS'])[0])
print('Pearson Correlation (HeavyAtomCount): ',pearsonr([d[descriptors.index('HeavyAtomCount')] for d in trainSet['Descriptors']],trainSet['logS'])[0])

#Load the test same using the same preparation steps as before.
testSet = pd.read_csv('test1.smi',encoding='iso-8859-1',na_values=['?'],delim_whitespace=True, names=['Number','ID','Name','logS','Detail','Detail2','Smiles'])[['Number','ID','Name','logS','Smiles']]
testSet.dropna(axis='index',inplace=True)
PandasTools.AddMoleculeColumnToFrame(frame=testSet, smilesCol='Smiles', molCol='Molecule')
testSet = testSet.ix[testSet['Molecule'].map(lambda x: x is not None)]
testSet['Descriptors'] = testSet['Molecule'].map(lambda x: computeDescriptors(x,calculator))
testSet = testSet.ix[testSet['Descriptors'].map(lambda x: x is not None)]
sns.violinplot(testSet['logS'],cut=0,scale="count", inner="quartile")
testSet.describe()

figure,(plt1,plt2,plt3) = pyplot.subplots(1,3)
figure.set_size_inches(15,15)
#simple linear regression
from sklearn import linear_model

simple_ext_prediction = simple_linearreg.predict(testSet['Descriptors'].tolist())
plt1.scatter(testSet['logS'],simple_ext_prediction)
plt1.set_aspect('equal')
plt1.set_title('Simple linear regression')
plt1.set_xlabel('Measured logS')
plt1.set_ylabel('Predicted logS')
plt1.set_xlim(-12,2)
plt1.set_ylim(-12,2)

#simple decision tree regression
simple_tree_ext_prediction = simple_tree.predict(testSet['Descriptors'].tolist())
plt2.scatter(testSet['logS'],simple_tree_ext_prediction)
plt2.set_aspect('equal')
plt2.set_title('Standard Decision Tree')
plt2.set_xlabel('Measured logS')
plt2.set_ylabel('Predicted logS')
plt2.set_xlim(-12,2)
plt2.set_ylim(-12,2)

#custom decision tree regression
custom_tree_ext_prediction = custom_tree.predict(testSet['Descriptors'].tolist())
plt3.scatter(testSet['logS'],custom_tree_ext_prediction)
plt3.set_aspect('equal')
plt3.set_title('Custom Decision Tree (regularized)')
plt3.set_xlabel('Measured logS')
plt3.set_ylabel('Predicted logS')
plt3.set_xlim(-12,2)
plt3.set_ylim(-12,2)

from sklearn.metrics import mean_squared_error
print('Coefficient of determination R^2 (LR): ',(simple_linearreg.score(testSet['Descriptors'].tolist(),testSet['logS'])))
print( 'MSE (LR): ',(mean_squared_error(testSet['logS'],simple_ext_prediction)))
print( 'Coefficient of determination R^2 (Default Tree): ',(simple_tree.score(testSet['Descriptors'].tolist(),testSet['logS'])))
print( 'MSE (Default Tree): ',(mean_squared_error(testSet['logS'],simple_tree_ext_prediction)))
print( 'Coefficient of determination R^2 (Custom Tree): ',(custom_tree.score(testSet['Descriptors'].tolist(),testSet['logS'])))
print( 'MSE (Custom Tree): ',(mean_squared_error(testSet['logS'],custom_tree_ext_prediction)))

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
params = {'max_depth':[2,5,10,20],'min_samples_split':[2,8,32,128],'min_samples_leaf':[1,2,5,10]}
cv = KFold(n=len(trainSet['Descriptors'].tolist()),n_folds=10,shuffle=True)
gs = GridSearchCV(simple_tree, params, cv=cv,verbose=1,refit=True)
gs.fit(trainSet['Descriptors'].tolist(), trainSet['logS'])
print('Best score: ',gs.best_score_)
print('Training set performance using best parameters ',gs.best_params_)
best_treemodel = gs.best_estimator_
figure,(plt1,plt2) = pyplot.subplots(1,2)
figure.set_size_inches(15,15)
#training set evaluation
best_tree_int_prediction = best_treemodel.predict(trainSet['Descriptors'].tolist())
plt1.scatter(trainSet['logS'],best_tree_int_prediction)
plt1.set_aspect('equal')
plt1.set_title('Optimized tree on training set')
plt1.set_xlabel('Measured logS')
plt1.set_ylabel('Predicted logS')
plt1.set_xlim(-12,2)
plt1.set_ylim(-12,2)

#test set evaluation
best_tree_ext_prediction = best_treemodel.predict(testSet['Descriptors'].tolist())
plt2.scatter(testSet['logS'],best_tree_ext_prediction)
plt2.set_aspect('equal')
plt2.set_title('Optimized tree on test set')
plt2.set_xlabel('Measured logS')
plt2.set_ylabel('Predicted logS')
plt2.set_xlim(-12,2)
plt2.set_ylim(-12,2)

print('Coefficient of determination R^2 (Internal): ',(best_treemodel.score(trainSet['Descriptors'].tolist(),trainSet['logS'])))
print('MSE (Internal): ',(mean_squared_error(trainSet['logS'],best_tree_int_prediction)))
print('Coefficient of determination R^2 (External): ',(best_treemodel.score(testSet['Descriptors'].tolist(),testSet['logS'])))
print('MSE (External): ',(mean_squared_error(testSet['logS'],best_tree_ext_prediction)))
fig,a = pyplot.subplots(1,1)
fig.set_size_inches(15,5)
createImportancePlot(a,descriptors,best_treemodel.feature_importances_,"Most important descriptors in the optimized tree")

from sklearn.linear_model import Lasso,LassoCV



from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso
best_lasso = LassoCV(normalize=True,max_iter=10000).fit(trainSet['Descriptors'].tolist(), trainSet['logS'])
figure,(plt1,plt2) = pyplot.subplots(1,2)
figure.set_size_inches(15,15)
#training set evaluation
best_int_prediction = best_lasso.predict(trainSet['Descriptors'].tolist())
plt1.scatter(trainSet['logS'],best_int_prediction)
plt1.set_aspect('equal')
plt1.set_title('Optimized LASSO on training set')
plt1.set_xlabel('Measured logS')
plt1.set_ylabel('Predicted logS')
plt1.set_xlim(-12,2)
plt1.set_ylim(-12,2)

#test set evaluation
best_ext_prediction = best_lasso.predict(testSet['Descriptors'].tolist())
plt2.scatter(testSet['logS'],best_ext_prediction)
plt2.set_aspect('equal')
plt2.set_title('Optimized LASSO on test set')
plt2.set_xlabel('Measured logS')
plt2.set_ylabel('Predicted logS')
plt2.set_xlim(-12,2)
plt2.set_ylim(-12,2)

print('Explained variance (Internal): ',(best_lasso.score(trainSet['Descriptors'].tolist(),trainSet['logS'])))
print('MSE (Internal): ',(mean_squared_error(trainSet['logS'],best_int_prediction)))
print('Explained variance (External): ',(best_lasso.score(testSet['Descriptors'].tolist(),testSet['logS'])))
print('MSE (External): ',(mean_squared_error(testSet['logS'],best_ext_prediction)))
fig,a = pyplot.subplots(1,1)
fig.set_size_inches(15,5)
createImportancePlot(a,descriptors,best_lasso.coef_,"Most important descriptors in the LASSO Regression")

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import ElasticNetCV
#params = {'alpha':[0.0001,0.001,0.01,0.1,1.0,2.0,4.0,10.0],'normalize':[True], 'warm_start':[False]}
#cv = KFold(n=len(trainSet['Descriptors'].tolist()),n_folds=10,shuffle=True)
#gs = GridSearchCV(Lasso(normalize=True), params, cv=cv,verbose=1,refit=True)
best_elastic = ElasticNetCV(normalize=True,max_iter=10000).fit(trainSet['Descriptors'].tolist(), trainSet['logS'])
#print('Training set performance using best parameters ',gs.best_params_)
#best_lasso = gs.best_estimator_
figure,(plt1,plt2) = pyplot.subplots(1,2)
figure.set_size_inches(15,15)
#training set evaluation
best_int_prediction = best_elastic.predict(trainSet['Descriptors'].tolist())
plt1.scatter(trainSet['logS'],best_int_prediction)
plt1.set_aspect('equal')
plt1.set_title('Optimized ElasticNet on training set')
plt1.set_xlabel('Measured logS')
plt1.set_ylabel('Predicted logS')
plt1.set_xlim(-12,2)
plt1.set_ylim(-12,2)

#test set evaluation
best_ext_prediction = best_elastic.predict(testSet['Descriptors'].tolist())
plt2.scatter(testSet['logS'],best_ext_prediction)
plt2.set_aspect('equal')
plt2.set_title('Optimized ElasticNet on test set')
plt2.set_xlabel('Measured logS')
plt2.set_ylabel('Predicted logS')
plt2.set_xlim(-12,2)
plt2.set_ylim(-12,2)

print('Explained variance (Internal): ',(best_elastic.score(trainSet['Descriptors'].tolist(),trainSet['logS'])))
print('MSE (Internal): ',(mean_squared_error(trainSet['logS'],best_int_prediction)))
print('Explained variance (External): ',(best_elastic.score(testSet['Descriptors'].tolist(),testSet['logS'])))
print('MSE (External): ',(mean_squared_error(testSet['logS'],best_ext_prediction)))
fig,a = pyplot.subplots(1,1)
fig.set_size_inches(15,5)
createImportancePlot(a,descriptors,best_elastic.coef_,"Most important descriptors in the ElasticNet Regression")

#normalize descriptors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(trainSet['Descriptors'].tolist())
std_patterns = scaler.transform(trainSet['Descriptors'].tolist())
std_test_patterns = scaler.transform(testSet['Descriptors'].tolist())

#compute pca; provide ten most descriptive principal components
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(std_patterns)
#print 'rel. explained variance per pc',pca.explained_variance_ratio_
pc_patterns = pca.transform(std_patterns)
pc_test = pca.transform(std_test_patterns)
fig,(plt1,plt2) = pyplot.subplots(2,1)
fig.set_size_inches(15,5)
fig.set_size_inches(15,10)
fig.subplots_adjust(hspace=0.5)
createImportancePlot(plt1,descriptors,pca.components_[0],"Most important descriptors in 1st principal component (Hypothesis: Size)")
createImportancePlot(plt2,descriptors,pca.components_[1],"Most important descriptors in 2nd principal component (Hypothesis: Electrostatic interactions)")

#evaluate modelling performance
figure,(plt1,plt2,plt3) = pyplot.subplots(1,3)
figure.set_size_inches(15,5)

#training
pc_linearreg = linear_model.LinearRegression()
pc_linearmodel = pc_linearreg.fit(pc_patterns,trainSet['logS'])
pc_prediction = pc_linearmodel.predict(pc_patterns)
plt1.scatter(trainSet['logS'],pc_prediction)
plt1.set_aspect('equal')
plt1.set_title('Full model regression')
plt1.set_xlabel('Measured logS')
plt1.set_ylabel('Predicted logS')
plt1.set_xlim(-12,2)
plt1.set_ylim(-12,2)

plt2.scatter(trainSet['logS'], [pc[0] for pc in pc_patterns])
plt2.set_title('1st PC')
plt2.set_xlabel('Measured logS')
plt2.set_ylabel('1st PC')


plt3.scatter(trainSet['logS'], [pc[1] for pc in pc_patterns])
plt3.set_title('2nd PC')
plt3.set_xlabel('Measured logS')
plt3.set_ylabel('2nd PC')


#test set
figure,(plt1,plt2,plt3) = pyplot.subplots(1,3)
figure.set_size_inches(15,5)

pc_prediction = pc_linearmodel.predict(pc_test)
plt1.scatter(testSet['logS'],pc_prediction)
plt1.set_aspect('equal')
plt1.set_title('Full model regression')
plt1.set_xlabel('Measured logS')
plt1.set_ylabel('Predicted logS')
plt1.set_xlim(-12,2)
plt1.set_ylim(-12,2)

plt2.scatter(testSet['logS'], [pc[0] for pc in pc_test])
plt2.set_title('1st PC')
plt2.set_xlabel('Measured logS')
plt2.set_ylabel('1st PC')


plt3.scatter(testSet['logS'], [pc[1] for pc in pc_test])
plt3.set_title('2nd PC')
plt3.set_xlabel('Measured logS')
plt3.set_ylabel('2nd PC')

print(pc_linearmodel.coef_)
fig,a = pyplot.subplots(1,1)
fig.set_size_inches(15,5)
createImportancePlot(a,['PC 1','PC 2'],pc_linearmodel.coef_,"PC Coefficients")









