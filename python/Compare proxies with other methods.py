get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Antoine' -i -v -m -p pandas,numpy,matplotlib,sklearn")

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
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import classification_report, confusion_matrix

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
y1 = df['SIO2(WT%)']
df['Na2O+K2O(Wt%)'] = df['NA2O(WT%)'] + df['K2O(WT%)']
y2 = df['Na2O+K2O(Wt%)']

cv = KFold(n_splits=10, shuffle=True)

##### predicting SiO2 using cross-val_predict
SVM = make_pipeline(StandardScaler(), SVR(gamma=0.1, C=100, cache_size=2000))
df['SiO2_pred'] = cross_val_predict(SVM, X, y1, cv=cv, n_jobs=3)

##### predicting SiO2 using cross-val_predict
SVM = make_pipeline(StandardScaler(), SVR(gamma=0.1, C=10, cache_size=2000))
df['Na2O+K2O_pred'] = cross_val_predict(SVM, X, y2, cv=cv, n_jobs=3)

###### TAS diagram
hor_ax = df['SiO2_pred']
vert_ax = df['Na2O+K2O_pred']
im = np.array(Image.open(r'C:\Users\Antoine Caté\Dropbox\Documents\Code\Geochem on python\TAS diagram 77-15.jpg'))
fig, ax = plt.subplots()
ax.imshow(im, extent=[35, 77, 0, 15], aspect='auto', cmap='gray')
ax.scatter(hor_ax, vert_ax)
ax.set(xlim=[35, 77], ylim=[0, 15])
plt.show()

###### Classify samples (from analysis)
def f_analysis(row):
    if row['SIO2(WT%)']<52:
        row['Rock_type_TAS']='basalt'
    elif row['SIO2(WT%)']<57:
        row['Rock_type_TAS']='basaltic-andesite'
    elif row['SIO2(WT%)']<63:
        row['Rock_type_TAS']='andesite'
    elif row['SIO2(WT%)'] + row['NA2O(WT%)'] + row['K2O(WT%)'] < 77:
        row['Rock_type_TAS']='dacite'
    else:
        row['Rock_type_TAS']='rhyolite'
    return row
df = df.apply(f_analysis, axis=1)

###### Classify samples (from prediction)
def f_analysis(row):
    if row['SiO2_pred']<52:
        row['Rock_type_pred']='basalt'
    elif row['SiO2_pred']<57:
        row['Rock_type_pred']='basaltic-andesite'
    elif row['SiO2_pred']<63:
        row['Rock_type_pred']='andesite'
    elif row['SiO2_pred'] + row['Na2O+K2O_pred'] < 77:
        row['Rock_type_pred']='dacite'
    else:
        row['Rock_type_pred']='rhyolite'
    return row
df = df.apply(f_analysis, axis=1)

###### TAS diagram plotting real rock type using analyses
groups = df.groupby('Rock_type_TAS')
im = np.array(Image.open(r'C:\Users\Antoine Caté\Dropbox\Documents\Code\Geochem on python\TAS diagram 77-15.jpg'))
fig, ax = plt.subplots()
ax.imshow(im, extent=[35, 77, 0, 15], aspect='auto', cmap='gray')
for name, group in groups:
    ax.plot(group['SIO2(WT%)'], group['Na2O+K2O(Wt%)'], marker='o', linestyle='', ms=3, label=name)
ax.legend()
ax.set(xlim=[35, 77], ylim=[0, 15], title='TAS diagram', xlabel='SiO2 (Wt%)', ylabel='Na2O + K2O (Wt%)')
plt.show()

###### TAS diagram plotting predicted rock type using analyses
groups = df.groupby('Rock_type_pred')
im = np.array(Image.open(r'C:\Users\Antoine Caté\Dropbox\Documents\Code\Geochem on python\TAS diagram 77-15.jpg'))
fig, ax = plt.subplots()
ax.imshow(im, extent=[35, 77, 0, 15], aspect='auto', cmap='gray')
for name, group in groups:
    ax.plot(group['SIO2(WT%)'], group['Na2O+K2O(Wt%)'], marker='o', linestyle='', ms=3, label=name)
ax.legend()
ax.set(xlim=[35, 77], ylim=[0, 15], title='TAS diagram', xlabel='SiO2 (Wt%)', ylabel='Na2O + K2O (Wt%)')
plt.show()

###### TAS diagram plotting real rock type using predicted concentrations
groups = df.groupby('Rock_type_TAS')
im = np.array(Image.open(r'C:\Users\Antoine Caté\Dropbox\Documents\Code\Geochem on python\TAS diagram 77-15.jpg'))
fig, ax = plt.subplots()
ax.imshow(im, extent=[35, 77, 0, 15], aspect='auto', cmap='gray')
for name, group in groups:
    ax.plot(group['SiO2_pred'], group['Na2O+K2O_pred'], marker='o', linestyle='', ms=3, label=name)
ax.legend()
ax.set(xlim=[35, 77], ylim=[0, 15], title='TAS diagram', xlabel='SiO2 predicted (Wt%)', ylabel='Na2O + K2O predicted (Wt%)')
plt.show()

###### TAS diagram plotting predicted rock type using analyses
groups = df.groupby('Rock_type_pred')
im = np.array(Image.open(r'C:\Users\Antoine Caté\Dropbox\Documents\Code\Geochem on python\TAS diagram 77-15.jpg'))
for name, group in groups:
    fig, ax = plt.subplots()
    ax.imshow(im, extent=[35, 77, 0, 15], aspect='auto', cmap='gray')
    ax.plot(group['SIO2(WT%)'], group['Na2O+K2O(Wt%)'], marker='o', linestyle='', ms=3, label=name)
    ax.legend()
    ax.set(xlim=[35, 77], ylim=[0, 15], title='TAS diagram of ' + name + ' with predicted concentrations', xlabel='SiO2 (Wt%)', ylabel='Na2O + K2O (Wt%)')
    plt.show()

List_litho = df['Rock_type_TAS'].unique()
#### Classification report
print(classification_report(df['Rock_type_TAS'], df['Rock_type_pred'],
                            labels=List_litho, target_names=List_litho))

#### Confusion matrix
conf_mat = confusion_matrix(df['Rock_type_TAS'], df['Rock_type_pred'], labels= List_litho)
conf_mat = pd.DataFrame(conf_mat, columns=List_litho, index=List_litho)
conf_mat.head(10)

###### Classify samples into Winch and Floyd categories (from analysis)
def f_analysis_Winch(row):
    if row['SIO2(WT%)']<52:
        row['Rock_type_WinchFromTAS']='basalt'
    elif row['SIO2(WT%)']<63:
        row['Rock_type_WinchFromTAS']='andesite_basaltic-andesite'
    else:
        row['Rock_type_WinchFromTAS']='dacite_rhyolite'
    return row
df = df.apply(f_analysis_Winch, axis=1)

###### Winchester and Floyd diagram using simplified categories from TAS
groups = df.groupby('Rock_type_WinchFromTAS')

im = np.array(Image.open(r'C:\Users\Antoine Caté\Dropbox\Documents\Code\Geochem on python\Winch and Floyd by Pearce.jpg'))
fig, ax = plt.subplots()
ax.imshow(im, extent=[0.01, 10, 0.001, 1], aspect='auto', cmap='gray')
for name, group in groups:
    ax.plot(group['NB(PPM)'] / group['Y(PPM)'],
            group['ZR(PPM)'] / (group['TIO2(WT%)']*10000/1.6681),
            marker='o', linestyle='', ms=3, label=name)
ax.set(xlim=[0.01, 10], ylim=[0.001, 1],
        title='TAS diagram of ' + name + ' with predicted concentrations',
       xlabel='Nb/Y', ylabel='Zr/Ti')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()


###### Winchester and Floyd diagram using categories from TAS
groups = df.groupby('Rock_type_TAS')

im = np.array(Image.open(r'C:\Users\Antoine Caté\Dropbox\Documents\Code\Geochem on python\Winch and Floyd by Pearce.jpg'))
fig, ax = plt.subplots()
ax.imshow(im, extent=[0.01, 10, 0.001, 1], aspect='auto', cmap='gray')
for name, group in groups:
    ax.plot(group['NB(PPM)'] / group['Y(PPM)'],
            group['ZR(PPM)'] / (group['TIO2(WT%)']*10000/1.6681),
            marker='o', linestyle='', ms=3, label=name)
ax.set(xlim=[0.01, 10], ylim=[0.001, 1],
        title='TAS diagram of ' + name + ' with predicted concentrations',
       xlabel='Nb/Y', ylabel='Zr/Ti')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()





