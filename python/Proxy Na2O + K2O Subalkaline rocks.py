get_ipython().magic('matplotlib inline')

import time

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

from PIL import Image

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error

pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)

xl_file = pd.ExcelFile('sub_alcaline_dataset_GeoRock.xlsx')
dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
sheet_names = list(dfs.keys())
df = dfs[sheet_names[0]]

######### Removing columns with too many missing data
a = df.iloc[:, 15:91].count(axis=0)
a = len(df.index) - a
a[a!=0]
df.drop(a[a>50].index.values, axis=1, inplace=True)
######## Removing samples with missing data
df.dropna(subset=a[a<=50].index.values, inplace=True)

######## Drop all useless columns
df.drop(df.iloc[:, 35:].columns.values, axis=1, inplace=True)

######## Removing all lines with values <= 0
df = df[~(df.iloc[:, 15:] <= 0).any(axis=1)]

print ('Shape of the df dataset after transformation:' + str(df.shape))
df.head()

####### Removing alkaline samples
df = df.ix[(df['SIO2(WT%)']>45)]
df = df.ix[(df['NA2O(WT%)']+df['K2O(WT%)']<5) | (df['SIO2(WT%)']>52)]
a = (5-2)/(52-35)
b = 2-35*a
df = df.ix[(df['SIO2(WT%)']<52) | (df['NA2O(WT%)']+df['K2O(WT%)']<a*df['SIO2(WT%)']+b)]


###### TAS diagram
hor_ax = df['SIO2(WT%)']
vert_ax = df['K2O(WT%)'] + df['NA2O(WT%)']
im = np.array(Image.open(r'C:\Users\Antoine Caté\Dropbox\Documents\Code\Geochem on python\TAS diagram 77-15.jpg'))
fig, ax = plt.subplots()
ax.imshow(im, extent=[35, 77, 0, 15], aspect='auto', cmap='gray')
ax.scatter(hor_ax, vert_ax)
ax.set(xlim=[35, 77], ylim=[0, 15])
plt.show()

def make_ALR(dataframe, element):
    temp_df = pd.DataFrame()
    for index, Series in dataframe.iterrows():
        Series = Series.apply(lambda x: x/Series[element])
        Series = np.log(Series)
        temp_df = temp_df.append(Series)
        temp_df.drop(element, axis=1, inplace=True)
    return temp_df

#### creating x and y
X = df.loc[:, ['TIO2(WT%)', 'AL2O3(WT%)', 'Y(PPM)', 'ZR(PPM)',
               'NB(PPM)', 'LA(PPM)', 'CE(PPM)', 'SM(PPM)', 'YB(PPM)', 'TH(PPM)']]
X = make_ALR(X, 'YB(PPM)')
y = df['NA2O(WT%)'] + df['K2O(WT%)']

#### creating test and train datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

C_range = 10.0 ** np.arange(-3, 3)
gamma_range= 10.0 ** np.arange(-3, 3)
param_grid = dict(svr__C=C_range, svr__gamma=gamma_range)
SVM = make_pipeline(StandardScaler(), SVR(cache_size=2000))
# scoring options: ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'log_loss', 'mean_squared_error', 'precision', 'r2', 'recall', 'roc_auc'] 
scores = ['r2']
print('Heat map:')
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print("Best parameters set found on development set:")
    grid = GridSearchCV(SVM, param_grid, cv=4, scoring='%s' % score, n_jobs=-1)
    #clf = grid.get_params()
    clf = grid.fit(X_train, y_train)
    print(clf.cv_results_['params'][clf.best_index_])
    print("Score: %0.03f" % clf.best_score_)
    print("")
    #for params, mean_score, scores in clf.grid_scores_:
         #print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    # Do a graphic representation
    scores = [x[1] for x in clf.grid_scores_]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))

from matplotlib.colors import Normalize


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

print('Heat map:')
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.3, right=0.8, bottom=0.3, top=0.8)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.4, midpoint=0.83))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()

####### Score on testing set
SVM = make_pipeline(StandardScaler(), SVR(gamma=0.1, C=100, cache_size=2000))
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
R2 = r2_score(y_test, y_pred)
Mean_error = mean_absolute_error(y_test, y_pred)
Median_error = median_absolute_error(y_test, y_pred)
Squared_error = mean_squared_error(y_test, y_pred)
print ('Scores obtained on the test dataset:\n R2: ' + str(R2) +
       '\n Mean absolute error:' + str(Mean_error) +
       '\n Median absolute error: ' + str(Median_error) + 
      '\n Mean squared error: ' + str(Squared_error))

from scipy.stats.stats import pearsonr
corr = pearsonr(y_test, y_pred)

fig, ax = plt.subplots()
line = ax.plot([0, 10], [0, 10], 'r', linewidth=4, ls='--')
ax.scatter(y_test, y_pred)
ax.set(xlim=[0, 10], ylim=[0, 10])
ax.set(title='Prediction vs analysis plot with the 1:1 line')
ax.set_xlabel('SiO2 analysis (wt%)')
ax.set_ylabel('SiO2 predicted (wt%)')
ax.text(4, 2, 'Pearson correlation coeficient: %.2f' % (corr[0]))
plt.show()

from sklearn.model_selection import cross_val_predict, KFold
cv = KFold(n_splits=10, shuffle=True)
y_pred = cross_val_predict(SVM, X, y, cv=cv, n_jobs=3)


from scipy.stats.stats import pearsonr
corr = pearsonr(y, y_pred)

fig, ax = plt.subplots()
line = ax.plot([0, 10], [0, 10], 'r', linewidth=4, ls='--')
ax.scatter(y, y_pred)
ax.set(xlim=[0, 10], ylim=[0, 10])
ax.set(title='Prediction vs analysis plot with the 1:1 line')
ax.set_xlabel('SiO2 analysis (wt%)')
ax.set_ylabel('SiO2 predicted (wt%)')
ax.text(4, 2, 'Pearson correlation coeficient: %.2f' % (corr[0]))
plt.show()



