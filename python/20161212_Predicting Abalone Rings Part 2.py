import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
get_ipython().magic('matplotlib inline')
abaloneDF = pd.read_csv('abalone.csv', names=['Sex', 'Length', 'Diameter', 'Height',
                                              'Whole Weight', 'Shucked Weight',
                                              'Viscera Weight', 'Shell Weight',
                                              'Rings'])
abaloneDF['Male'] = (abaloneDF['Sex'] == 'M').astype(int)
abaloneDF['Female'] = (abaloneDF['Sex'] == 'F').astype(int)
abaloneDF['Infant'] = (abaloneDF['Sex'] == 'I').astype(int)
abaloneDF = abaloneDF[abaloneDF['Height'] > 0]

dataset = abaloneDF.drop(['Rings', 'Sex'],axis=1)
#We will start with the same number of components as variables
pca_model = PCA(n_components=10)
pca_model.fit(dataset)
#Plot the explained variance
plt.plot(range(1,11),pca_model.explained_variance_ratio_);
plt.xlabel('Principal Component');
plt.ylabel('Percentage Explained Variance');

df = pd.DataFrame(data=pca_model.components_)
df.index = dataset.columns
dfRed = df.ix[:,0:3]
dfRed.columns = range(1,5)
dfRed.plot.bar();
plt.ylabel('Coefficient');

#Remove the last 6 PCs
red_PCA = PCA(n_components=4)
red_PCA.fit(dataset)
rings = abaloneDF['Rings'].values.reshape(len(abaloneDF),1)
red_data = np.hstack([red_PCA.transform(dataset),rings])
red_df = pd.DataFrame(red_data,columns=['PC1','PC2','PC3','PC4','Rings'])
train, test = train_test_split(red_df,train_size=0.7)
xtrain = train.drop(['Rings'],axis=1)
ytrain = train['Rings']
xtest = test.drop(['Rings'],axis=1)
ytest = test['Rings']
regr = linear_model.LinearRegression()
regr.fit(xtrain, ytrain)
#Take a look at the regression coefficients
dict(zip(list(xtrain.columns),regr.coef_))

#Same function as in Part 1:
def plot_yyhat(ytest,ypred):
    r2 = r2_score(ytest, ypred )
    mae = mean_absolute_error(ytest, ypred)
    absmin = min([ytest.min(),ypred.min()])
    absmax = max([ytest.max(),ypred.max()])
    ax = plt.axes()
    ax.scatter(ytest,ypred)
    ax.set_title('Y vs. YHat')
    ax.axis([absmin, absmax, absmin, absmax])
    ax.plot([absmin, absmax], [absmin, absmax],c="k")
    ax.set_ylabel('Predicted Rings') 
    ax.set_xlabel('Actual Rings')
    #Plot the text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textStr = '$MAE=%.3f$\n$R2=%.3f$' % (mae, r2)
    ax.text(0.05, 0.95, textStr, transform=ax.transAxes, fontsize=14,
               verticalalignment='top', bbox=props);
ypred = regr.predict(xtest)
plot_yyhat(ytest,ypred)

