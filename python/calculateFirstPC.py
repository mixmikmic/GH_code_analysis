import pandas as pd
import numpy as np
import datetime
from sklearn.decomposition import PCA

def getFirstPC(df):
    df_norm = (df - df.mean()) / (df.std()) # normalize data to the same unit
    cv = df_norm.cov()
    xcv, vcv = np.linalg.eig(cv)
    pc = df_norm.dot(-vcv).iloc[:, 0]
    return pc

def getFirstPComp(df):
    pca = PCA(n_components=1)
    pca.fit(df)
    pc = df.dot(pca.components_.T)
    return pc

# for tech
path = r'linear forecasting\data\\'
tech = pd.read_csv(path+r'PCAtech.csv',index_col=0,usecols=np.arange(0,14))
tech_pc = getFirstPC(tech)
tech['PCAtech'] = tech_pc
tech.to_csv(path+r'PCAtech.csv')

# for price

# read raw data
varNames = ['DP','PE','BM','CAPE']
DP = pd.read_csv(path + varNames[0]+'.csv', index_col=0, parse_dates=[0]) # montly at end of month
BM = pd.read_csv(path + varNames[2]+'.csv', index_col=0, parse_dates=[0]) # monthly at end of month
PE = pd.read_csv(path + varNames[1]+'.csv',parse_dates=[0])# monthly at beginning of month
CAPE = pd.read_csv(path + varNames[3]+'.csv', parse_dates=[0]) # monthly at beginning of month

# matching dates
CAPE['Date'] = CAPE['Date'] - datetime.timedelta(1)
PE['Date'] = PE['Date'] - datetime.timedelta(1)
CAPE = CAPE.set_index('Date')
PE = PE.set_index('Date')

# join four prices 
price = pd.concat([DP,PE,BM,CAPE],axis=1)
price = price.dropna(axis=0, how='any')
price.head()

# pca
price_pc = getFirstPC(price)
price['PCAprice'] = price_pc
price_pc.describe()
price.to_csv(path+r'PCAprice.csv')





