get_ipython().magic('pylab inline')
import pandas as pd
from copy import deepcopy as copy

iev = pd.read_excel('../data/iev.xlsx', sheetname='DATASET')
iev.columns = ['cve','bugname','cveDate','minVersion','maxVersion']
iev['product']='ie'
print iev.shape
iev.head(2)

asv = pd.read_excel('../data/asv.xlsx', sheetname='DATASET')
asv.columns = ['cve','bugname','cveDate','minVersion','maxVersion']
asv['product']='safari'
print asv.shape
asv.head(2)

ffv = pd.read_excel('../data/ffv.xlsx', sheetname='DATASET')
ffv.drop(['mfsa'], axis=1, inplace=True)
ffv.columns = ['bugname','cve','bugDate','cveDate','minVersion','maxVersion']
ffv['product']='ff'
print ffv.shape
ffv.head(2)

gcv = pd.read_excel('../data/gcv.xlsx', sheetname='DATASET')
gcv.drop(['codeMinVer','codeMaxVer'], axis=1, inplace=True)
gcv.columns = ['bugname','cve','cveDate','bugDate','minVersion','maxVersion']
gcv['product']='chrome'
print gcv.shape
gcv.head(2)

vulns = pd.concat([gcv, ffv, asv, iev])
vulns.dropna(how='any',subset=['maxVersion', 'minVersion'], axis=0, inplace=True)

vuln_list = []
for i, row in vulns.iterrows(): #slowish, but whatever
    for version in range(int(row['minVersion']), int(row['maxVersion']+1)):
        row['version'] = version
        vuln_list.append(copy(row))
        
vulns = pd.DataFrame(vuln_list)
vulns.drop(['maxVersion', 'minVersion'], axis=1, inplace=True)

vulns.sort(columns=['product', 'version', 'cve'], inplace=True)
print vulns.shape
vulns.head(3)

# corresponds to the NVD data set (to the best of my understanding)
NVD = vulns[['cveDate','cve','product','version']].drop_duplicates()
NVD.set_index(['product', 'version'], inplace=True)
NVD_counts = NVD.groupby(level=[0,1]).apply(lambda x: x.set_index('cveDate').resample('M', how='count').reset_index())
NVD_cumulative = NVD_counts['cve'].unstack().T.cumsum().T
NVD_cumulative.columns = NVD_cumulative.columns + 1
NVD_cumulative.insert(0, 0, 0, allow_duplicates=False)
print NVD_cumulative.shape
NVD_cumulative.head(2)

NVD_cumulative.columns + 1

# corresponds to the union of the NVD.bug and NVD.advice datasets
NVD_Bug = vulns.dropna(subset=['bugname'])[['cveDate','cve','product','version']].drop_duplicates()
NVD_Bug.set_index(['product', 'version'], inplace=True)
NVD_Bug_counts = NVD_Bug.groupby(level=[0,1]).apply(lambda x: x.set_index('cveDate').resample('M', how='count').reset_index())
NVD_Bug_cumulative = NVD_Bug_counts['cve'].unstack().T.cumsum().T
NVD_Bug_cumulative.columns = NVD_Bug_cumulative.columns + 1
NVD_Bug_cumulative.insert(0, 0, 0, allow_duplicates=False)
print NVD_Bug_cumulative.shape
NVD_Bug_cumulative.head(2)

# Corresponds to the NVD.Nbug dataset (to the best of my understanding)
Bug = vulns.dropna(subset=['bugname'])[['cveDate','cve','product','version']]
Bug.set_index(['product', 'version'], inplace=True)
Bug_counts = Bug.groupby(level=[0,1]).apply(lambda x: x.set_index('cveDate').resample('M', how='count').reset_index())
Bug_cumulative = Bug_counts['cve'].unstack().T.cumsum().T
Bug_cumulative.columns = Bug_cumulative.columns + 1
Bug_cumulative.insert(0, 0, 0, allow_duplicates=False)
Bug_cumulative.head(2)

# Corresponds to the Advice.Nbug dataset (to the best of my understanding)
Bug_date = vulns[['bugDate','cve','product','version']].dropna(subset=['bugDate'])
Bug_date.set_index(['product', 'version'], inplace=True)
Bug_date_counts = Bug_date.groupby(level=[0,1]).apply(lambda x: x.set_index('bugDate').resample('M', how='count').reset_index())
Bug_date_cumulative = Bug_date_counts['cve'].unstack().T.cumsum().T
Bug_date_cumulative.columns = Bug_date_cumulative.columns + 1
Bug_date_cumulative.insert(0, 0, 0, allow_duplicates=False)
Bug_date_cumulative.head(2)

datasets = pd.concat([NVD_cumulative, 
                     NVD_Bug_cumulative, 
                     Bug_cumulative, 
                     Bug_date_cumulative],
                    keys=['NVD',
                          'NVD.Bug', 
                          'NVD.Advice-Bug', 
                          'Advice.Nbug'])

obs_samples_list = []
for idx, row in datasets.iterrows(): #slowish, but whatever
    Tmax = row.notnull().sum()
    if Tmax >= 6: #need 6 months of data, plus month 0
        for t in range(6,Tmax):
            sample = row.loc[:t]
            sample['DS'] = idx[0] #there is probably a better way to do this
            sample['product'] = idx[1]
            sample['version'] = idx[2]
            sample['tseries'] = t
            obs_samples_list.append(sample)
        
obs_samples = pd.DataFrame(obs_samples_list)
obs_samples.set_index(['DS','product','version','tseries'], inplace=True)
print obs_samples.shape
obs_samples.head()

obs_samples.to_pickle('_!obs_samples!_.pickle')



