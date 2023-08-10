# numerics
import numpy as np
import pandas as pd
from scipy import stats

# learn you some machines
from sklearn import linear_model

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

# viz
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns

# essentials
from pprint import pprint
import os, io, re
from datetime import datetime

# Read water treatment plant data into a dataframe
df = pd.read_csv('data/water_treatment/water-treatment.data',
                    na_values='?',
                    header=None)

# Load information about column codes and their descriptions
info = pd.read_csv('data/water_treatment/water-treatment.names',
                    engine='python',
                    skiprows=64, nrows=38, 
                    names = ['Code','Description'],
                    header=None, 
                    index_col=0,
                    na_values='?',
                    delimiter=r' {2,}')

inputs      = info[  info['Description'].apply(lambda x : '(input' in x)  ]

plant_inputs      = info[  info['Description'].apply(lambda x : '(input' in x and 'plant' in x)  ]
tank1_inputs      = info[  info['Description'].apply(lambda x : '(input' in x and 'primary settler' in x)  ]
tank2_inputs      = info[  info['Description'].apply(lambda x : '(input' in x and 'secondary settler' in x)  ]
outputs     = info[  info['Description'].apply(lambda x : '(output' in x)   ]

perf_inputs = info[  info['Description'].apply(lambda x : '(performance input' in x)  ]
glob_inputs = info[  info['Description'].apply(lambda x : '(global performance input' in x)  ]

# Column labes are the codes, plus one more column for "Date"
column_labels = np.concatenate([np.array(['Date']), info['Code'].values])
df.columns = column_labels

# Parse the Date column from D-(day)/(month)/(yr) to datetime
df['Date'] = df['Date'].apply(lambda x : datetime.strptime(x,'D-%d/%m/%y'))

# Use the date as the index
df.set_index('Date')

print("")

# color by label suffix (plant input, primary input, secondary input)
def get_cmap(n):
    #colorz = plt.cm.cool
    colorz = plt.get_cmap('Set1')
    return [ colorz(float(i)/n) for i in range(n)]

colorz = get_cmap(4)

zippy = np.linspace(0,1000,5)
plt.plot(zippy,zippy,'k-')
plt.scatter(df['SS-E'].values, df['SS-P'].values, alpha=0.4, color=colorz[0], label='Plant In/Primary')
plt.scatter(df['SS-P'].values, df['SS-D'].values, alpha=0.4, color=colorz[1], label='Primary/Secondary')
plt.scatter(df['SS-D'].values, df['SS-S'].values, alpha=0.4, color=colorz[2], label='Secondary/Outlet')
plt.title("Ratios: Outlet (y) vs. Inlet (x)")
plt.legend()
plt.show()

def plot_block_input_output_regression(lab1, lab2, mycolor):
    # use scikit-learn linear_model
    # build linear model of input vs. output
    lmreg = linear_model.LinearRegression( fit_intercept = False )
    dat = df[[lab1, lab2]].dropna()
    lmreg.fit( dat[lab1].values.reshape(-1,1), dat[lab2].values.reshape(-1,1) )
    
    xx = np.linspace(0,1300,100)
    yy = lmreg.predict(xx.reshape(-1,1))

    fig = plt.figure(figsize=(6,5))
    
    ax1 = fig.add_subplot(111)
    
    zippy = np.linspace(0,1000,5)
    ax1.plot(zippy,zippy,'k-')
    ax1.scatter(df[lab1].values, df[lab2].values, alpha=0.4, color=mycolor, label=lab1+" v "+lab2)
    ax1.plot(xx, yy, '-', color=mycolor, label='Linear Regression')
    
    ax1.set_xlabel(lab1)
    ax1.set_ylabel(lab2)
    ax1.legend(loc='upper left')

plot_block_input_output_regression('SS-E','SS-P',colorz[0])
gca().set_xlim([0,1800])
gca().set_ylim([0,1800])

plot_block_input_output_regression('SS-P','SS-D',colorz[1])
gca().set_xlim([0,1300])
gca().set_ylim([0,1300])

plot_block_input_output_regression('SS-D','SS-S',colorz[2])
gca().set_xlim([0,300])
gca().set_ylim([0,300])

plt.show()

def plot_input_output_qq(lab1,lab2, mycolor):
    
    # use scikit-learn linear_model
    # build linear model of input vs. output
    ###lab1 = 'SS-E'
    ###lab2 = 'SS-P'
    lmreg = linear_model.LinearRegression( fit_intercept = False )
    dat = df[[lab1, lab2]].dropna()
    lmreg.fit( dat[lab1].values.reshape(-1,1), dat[lab2].values.reshape(-1,1) )

    xx = dat[[lab1,lab2]].dropna()[lab1].values.reshape(-1,1)
    yyhat = lmreg.predict( xx )
    yy = df[[lab1,lab2]].dropna()[lab1].shape

    resid = yyhat - yy
    resid = resid.reshape(len(resid),)

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)

    stats.probplot(resid, dist='norm', plot=ax)
    
    ax.get_lines()[0].set_color(mycolor)
    ax.get_lines()[1].set_color(mycolor)
    
    plt.title("Quantile-Quantile Plot: "+lab1+" vs "+lab2+" Residual",size=14)
    
    plt.show()
    
plot_input_output_qq('SS-E','SS-P',colorz[0])
plot_input_output_qq('SS-P','SS-D',colorz[1])
plot_input_output_qq('SS-D','SS-S',colorz[2])

def get_lm_resid(lab1, lab2, df):
    # use scikit-learn linear_model
    # build linear model of input vs. output
    df_resid = df[['Date',lab1,lab2]].dropna()

    xx = df_resid[lab1].values.reshape(-1,1)
    yy = df_resid[lab2].values.reshape(-1,1)

    lmreg = linear_model.LinearRegression( fit_intercept = False )
    lmreg.fit(xx,yy)
    yyhat = lmreg.predict(xx)

    resid = yy - yyhat
    resid = resid.reshape(len(resid),)
    df_resid['Residual'] = resid
    
    return df_resid

# use scikit-learn linear_model
# build linear model of input vs. output
lab1 = 'SS-E'
lab2 = 'SS-P'
df_residEP = get_lm_resid(lab1, lab2, df)
sns.distplot(df_residEP['Residual'].abs(),
             label=lab1+' vs '+lab2,
             kde_kws={'kernel':'tri'})


lab1 = 'SS-P'
lab2 = 'SS-D'
df_residPD = get_lm_resid(lab1, lab2, df)
sns.distplot(df_residPD['Residual'].abs(),
             label=lab1+' vs '+lab2,
             kde_kws={'kernel':'tri'})


lab1 = 'SS-D'
lab2 = 'SS-S'
df_residDS = get_lm_resid(lab1, lab2, df)
sns.distplot(df_residDS['Residual'].abs(),
             label=lab1+' vs '+lab2,
             kde_kws={'kernel':'tri'})


plt.legend()
plt.title('Distribution of L1 Norm of Residuals', size=14)
plt.show()

normal_x = df_residEP['Date'][df_residEP['Residual'].abs()<100]
normal_y = df_residEP['SS-E'][df_residEP['Residual'].abs()<100]

irregular_x = df_residEP['Date'][df_residEP['Residual'].abs()>100]
irregular_y = df_residEP['SS-E'][df_residEP['Residual'].abs()>100]

plt.plot(normal_x,normal_y,'o',color=colorz[0])
plt.plot(irregular_x,irregular_y,'bo',label="Outliers")

plt.title("Time Series: Suspsended Solids Plant Input\nOutliers Identified with Linear Regression",size=14)
plt.xlabel('Date')
plt.ylabel('SS-E Value')
plt.legend()

plt.show()

normal_x = df_residEP['Date'][df_residEP['Residual'].abs()<100]
normal_y = df_residEP['SS-P'][df_residEP['Residual'].abs()<100]

irregular_x = df_residEP['Date'][df_residEP['Residual'].abs()>100]
irregular_y = df_residEP['SS-P'][df_residEP['Residual'].abs()>100]

plt.plot(normal_x,normal_y,'o',color=colorz[1])
plt.plot(irregular_x,irregular_y,'bo',label="Outliers")

plt.title("Time Series: Suspsended Solids Primary Tank In\nOutliers Identified with Linear Regression",size=14)
plt.xlabel('Date')
plt.ylabel('SS-P Value')
plt.legend()

plt.show()

def make_marked_outlier_timeseries(df_resid, lab1, lab2, normal_cutoff, mycolor1, myoutliercolor1, mycolor2, myoutliercolor2):
    # use scikit-learn linear_model
    # build linear model of input vs. output
    df_resid = df[['Date',lab1,lab2]].dropna()

    xx = df_resid[lab1].values.reshape(-1,1)
    yy = df_resid[lab2].values.reshape(-1,1)
    
    # linear_model is the scikit learn module
    lmreg = linear_model.LinearRegression( fit_intercept = False )
    
    lmreg.fit(xx,yy)
    yyhat = lmreg.predict(xx)

    resid = yy - yyhat
    resid = resid.reshape(len(resid),)
    df_resid['Residual'] = resid
    
    
    
    fig = plt.figure(figsize=(10,8))
    ax1, ax2 = [fig.add_subplot(211+k) for k in range(2)]
    
    
    
    normal_x = df_resid['Date'][df_resid['Residual'].abs()<normal_cutoff]
    normal_y = df_resid[lab1][df_resid['Residual'].abs()<normal_cutoff]

    irregular_x = df_resid['Date'][df_resid['Residual'].abs()>normal_cutoff]
    irregular_y = df_resid[lab1][df_resid['Residual'].abs()>normal_cutoff]

    ax1.plot(normal_x,normal_y,'o',color=mycolor1)
    ax1.plot(irregular_x,irregular_y,'o',color=myoutliercolor1,label="Outliers")

    ax1.set_title("Time Series: "+lab1+"\nOutliers Identified with Linear Regression",size=14)
    ax1.set_xlabel("Date")
    ax1.set_ylabel(lab1+" Value")
    ax1.legend()
    
    
    # ------
    
    
    normal_x = df_resid['Date'][df_resid['Residual'].abs()<normal_cutoff]
    normal_y = df_resid[lab2][df_resid['Residual'].abs()<normal_cutoff]

    irregular_x = df_resid['Date'][df_resid['Residual'].abs()>normal_cutoff]
    irregular_y = df_resid[lab2][df_resid['Residual'].abs()>normal_cutoff]

    ax2.plot(normal_x,normal_y,'o',color=mycolor2)
    ax2.plot(irregular_x,irregular_y,'o',color=myoutliercolor2,label="Outliers")

    ax2.set_title("Time Series: "+lab2+"\nOutliers Identified with Linear Regression",size=14)
    ax2.set_xlabel("Date")
    ax2.set_ylabel(lab2+" Value")
    ax2.legend()

    
    fig.subplots_adjust(hspace = 0.5)
    
    plt.show()

make_marked_outlier_timeseries(df_residEP,'SS-E','SS-P',150,colorz[0],'b',colorz[1],'b')

make_marked_outlier_timeseries(df_residPD,'SS-P','SS-D',75,colorz[1],'b',colorz[2],'b')

make_marked_outlier_timeseries(df_residDS,'SS-D','SS-S',50,colorz[2],'b',colorz[3],'b')

all_labels = ['Date'] + inputs['Code'].tolist() + outputs['Code'].tolist()
nodate_labels = inputs['Code'].tolist() + outputs['Code'].tolist()
df_io = df[all_labels]

print df_io.shape
print df_io.dropna().shape
print 1.0*df_io.dropna().shape[0]/df_io.shape[0]

df_io.isnull().sum()

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values=np.nan,strategy='mean')
df_scrub = imp.fit_transform(df_io[nodate_labels])
print df_scrub

from numpy.linalg import inv

# Note that np.dot(A,B) does matrix multiplication if A and B are matrices

X = df_scrub
ii = inv( np.dot( X.T, X ) )
Hp = np.dot( X, ii )
H = np.dot( Hp, X.T )

print H.shape

n = X.shape[0]
p = X.shape[1]
threshold = ((2.0*p)/n)
Hdiag = np.diag(H)
sns.barplot(range(len(Hdiag)), Hdiag)
gca().plot(range(len(Hdiag)), threshold*np.ones(len(Hdiag)), 'k--')
gca().set_xticklabels('')
gca().set_title("Hat Matrix: Diagonal")
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

sns.heatmap(H[:,:], 
            annot=False, square=True,
            vmin=0.0,vmax=threshold, cmap=plt.cm.Purples,
            ax=ax)
ax.set_xticklabels('')
ax.set_yticklabels('')
plt.title("H Matrix Visualization", size=14)
plt.show()

print df_io.shape
print H.shape

#def make_hatmatrix_outlier_timeseries(df, lab1, lab2, normal_cutoff, mycolor1, myoutliercolor1, mycolor2, myoutliercolor2):
def make_hatmatrix_outlier_timeseries(df, lab1, normal_cutoff, mycolor1, myoutliercolor1):
    df.loc[:,'h_ii'] = np.diag(H)
    
    
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)

    
    normal_x = df['Date'][df['h_ii']<normal_cutoff]
    normal_y = df[lab1][df['h_ii']<normal_cutoff]
    irregular_x = df['Date'][df['h_ii']>normal_cutoff]
    irregular_y = df[lab1][df['h_ii']>normal_cutoff]
    
    ax1.plot(normal_x, normal_y, 'o', color=mycolor1)
    ax1.plot(irregular_x, irregular_y, 'o', color=myoutliercolor1, label="Outliers")
    
    ax1.set_title("Time Series: "+lab1+"\nOutliers Identified with Hat Matrix",size=14)
    ax1.set_xlabel("Date")
    ax1.set_ylabel(lab1+" Value")
    ax1.legend()

    
    
    '''
    normal_x = df['Date'][df['h_ii']<normal_cutoff]
    normal_y = df[lab2][df['h_ii']<normal_cutoff]
    irregular_x = df['Date'][df['h_ii']>normal_cutoff]
    irregular_y = df[lab2][df['h_ii']>normal_cutoff]
    
    ax2.plot(normal_x, normal_y, 'o', color=mycolor2)
    ax2.plot(irregular_x, irregular_y, 'o', color=myoutliercolor2, label="Outliers")
    
    ax2.set_title("Time Series: "+lab2+"\nOutliers Identified with Hat Matrix",size=14)
    ax2.set_xlabel("Date")
    ax2.set_ylabel(lab2+" Value")
    ax2.legend()
    '''

    
    fig.subplots_adjust(hspace = 0.5)

    plt.show()

make_hatmatrix_outlier_timeseries(df_io, 'SS-E', threshold, colorz[0], 'b')

make_hatmatrix_outlier_timeseries(df_io, 'SS-P', threshold, colorz[1], 'b')

make_hatmatrix_outlier_timeseries(df_io, 'SS-D', threshold, colorz[2], 'b')

make_hatmatrix_outlier_timeseries(df_io, 'SS-S', threshold, colorz[3], 'b')

make_hatmatrix_outlier_timeseries(df_io, 'SED-E', threshold, colorz[0], 'b')
make_hatmatrix_outlier_timeseries(df_io, 'SED-P', threshold, colorz[1], 'b')
make_hatmatrix_outlier_timeseries(df_io, 'SED-D', threshold, colorz[2], 'b')
make_hatmatrix_outlier_timeseries(df_io, 'SED-S', threshold, colorz[3], 'b')

pprint(info['Description'].tolist())













MDs = []
n = X.shape[0]
p = X.shape[1]
for h_ii in Hdiag:
    md = (n-1.0)*(h_ii - (1.0/n))
    MDs.append( md )

print X.shape
print info.shape

print info



















