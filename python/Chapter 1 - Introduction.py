import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

wage = pd.read_csv('../../data/Wage.csv')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

fig1 = sns.regplot(x='age', y='wage', data=wage, lowess=True, scatter_kws={'color': 'gray'}, ax=ax1)
fig1.set(xlabel='Age', ylabel='Wage')

fig2 = sns.regplot(x=wage['year'], y=wage['wage'], scatter_kws={'color': 'gray'}, ax=ax2)
fig2.set(xlim=(wage['year'].min()-1, wage['year'].max()+1), xlabel='Year', ylabel='Wage')

#need to extract the integer for education level with a lambda function 
wage['education_int'] = wage['education'].apply(lambda x: int(x.split('.')[0]))
fig3 = sns.boxplot(x=wage['education_int'], y=wage['wage'])
fig3.set(xlabel='Education Level', ylabel='Wage');

smarket = pd.read_csv('../../data/Smarket.csv')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

fig1 = sns.boxplot(x=smarket['Direction'], y=smarket['Lag1'], order=['Down', 'Up'], ax=ax1)
fig1.set(ylabel='Percentage Change in S&P', xlabel='Today\'s Direction', title='Yesterday');

fig2 = sns.boxplot(x=smarket['Direction'], y=smarket['Lag2'], order=['Down', 'Up'], ax=ax2)
fig2.set(ylabel='Percentage Change in S&P', xlabel='Today\'s Direction', title='Two Days Previous');

fig3 = sns.boxplot(x=smarket['Direction'], y=smarket['Lag3'], order=['Down', 'Up'], ax=ax3)
fig3.set(ylabel='Percentage Change in S&P', xlabel='Today\'s Direction', title='Three Days Previous');

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

nci60_x = pd.read_csv('../../data/NCI60_X.csv', index_col=0)
nci60_y = pd.read_csv('../../data/NCI60_y.csv')

#get the first two principal components
pca = PCA(n_components=2)
pca.fit(nci60_x)
pca_comps = pca.transform(nci60_x)

km = KMeans(n_clusters=4)
y_pred = km.fit_predict(pca_comps)

#not quite sure which algorithm was used to cluster the data in figure 1.4...
#I tried kmeans, gaussian mixture, and heirarchical clustering but none were comparable
#sticking with kmeans even though it's different
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(pca_comps[y_pred==0, 0], pca_comps[y_pred==0, 1], c='red')
ax1.scatter(pca_comps[y_pred==1, 0], pca_comps[y_pred==1, 1], c='blue')
ax1.scatter(pca_comps[y_pred==2, 0], pca_comps[y_pred==2, 1], c='green')
ax1.scatter(pca_comps[y_pred==3, 0], pca_comps[y_pred==3, 1], c='cyan')
ax1.set(xlabel='Z1', ylabel='Z2')

#not using different symbols for the second figure, just scaling the rainbow colormap
from matplotlib import colors, cm

uniq = list(nci60_y['x'].unique())
c_norm = colors.Normalize(vmin=0, vmax=len(uniq))
scalar_map = cm.ScalarMappable(norm=c_norm, cmap=plt.get_cmap('rainbow'))
for i in range(len(uniq)):
    ax2.scatter(pca_comps[nci60_y['x'] == uniq[i], 0], pca_comps[nci60_y['x'] == uniq[i], 1],
               color=scalar_map.to_rgba(i))
ax2.set(xlabel='Z1', ylabel='Z2')



