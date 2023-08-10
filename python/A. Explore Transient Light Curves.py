DATA_PATH = '../../data/'
LIGHTCURVES_PATH = DATA_PATH + 'lightcurves/'

import pandas as pd

filename = 'transient_catalog.pickle'
indir = DATA_PATH; filepath = indir + filename
df_cat = pd.read_pickle(filepath)
df_cat = pd.read_pickle(filepath)
print(df_cat.TransientID.unique().shape)

df_cat.head()

df_cat[df_cat.TransientID==1306151290014118570]

filename = 'transient_lightcurves.pickle'
indir = LIGHTCURVES_PATH; filepath = indir + filename
df_lcs = pd.read_pickle(filepath)
df_lcs['TransientID'] = pd.to_numeric(df_lcs['TransientID'].str[6:])
print(df_lcs.TransientID.unique().shape)

all_exist = True
transID_cat_list = df_cat.TransientID.unique()
for lcs_id in df_lcs.TransientID.unique():
    all_exist = all_exist and (lcs_id in transID_cat_list)
print('All exist:', str(all_exist))

all_exist = True
lcs_missing_transientID_list = []
transID_lcs_list = df_lcs.TransientID.unique()
for cat_id in df_cat.TransientID.unique():
    id_exists = (cat_id in transID_lcs_list)
    all_exist = all_exist and id_exists
    if not id_exists: lcs_missing_transientID_list.append(cat_id)
print('All exist:', str(all_exist))
print('Missing: {} Transients'.format(len(lcs_missing_transientID_list)))

df_merge = df_cat.copy().merge(df_lcs.copy().groupby('TransientID',as_index=False).count(), how='inner')
df_merge.rename(columns={'Mag':'ObsCount'}, inplace=True)
df_merge = df_merge[['TransientID', 'Classification', 'ObsCount']]

print(df_merge.shape)

df = df_merge[['Classification','ObsCount']].groupby('Classification').count()
df = df.rename(columns={'ObsCount':'ObjCount'}).sort_values('ObjCount', ascending=False)
df.head(10).transpose()

df_lcs.sort_values(['MJD'])['MJD'].iloc[0]

df_lcs.groupby('TransientID').count().describe()

df_merge_filtered = df_merge[df_merge.ObsCount >= 5]
df_merge_filtered.shape[0]

df = df_merge_filtered[['Classification','ObsCount']].groupby('Classification').count()
df = df.rename(columns={'ObsCount':'ObjCount'}).sort_values('ObjCount', ascending=False)
df.head(20).transpose()

df_lcs[df_lcs.TransientID.isin(df_merge_filtered.TransientID)].groupby('TransientID').count().describe()

df_merge_filtered = df_merge[df_merge.ObsCount >= 10]
df_merge_filtered.shape[0]

df = df_merge_filtered[['Classification','ObsCount']].groupby('Classification').count()
df = df.rename(columns={'ObsCount':'ObjCount'}).sort_values('ObjCount', ascending=False)
df.head(20).transpose()

df_lcs[df_lcs.TransientID.isin(df_merge_filtered.TransientID)].groupby('TransientID').count().describe()

def lightcurve(transID):
    df_lc = df_lcs[df_lcs.TransientID == transID]
    return df_lc

def class_random_ids(klass):
    df_class = df_merge[df_merge.Classification == klass]
    df_class = df_class[(df_class.ObsCount <= 40) & (df_class.ObsCount >= 10)]
    IDs = df_class.TransientID.unique()
    np.random.seed(40)
    rand = np.random.randint(0, IDs.shape[0]-1, 4)
    randIds = IDs[rand]
    return randIds

def plot_lightcurve(lc):
    plot = lc.plot(x='MJD', y='Mag', marker='.', markeredgecolor='black', linestyle='None', legend=False, figsize=(7,7),ylim=(0,30),  yerr='Magerr', elinewidth=0.7)
    plot.set_xlabel('Modified Julian Date')
    plot.set_ylabel('Mag')

top_classes = ['SN', 'CV', 'AGN', 'Blazar']

for i, klass in enumerate(top_classes):
    ids = class_random_ids(klass)
    for idx in ids:
        lc = lightcurve(idx)
        plot_lightcurve(lc)
        plt.savefig('graphs/{}_{}.png'.format(klass, idx))
        plt.close()

classified_ids = [1111031520324144272]
for ix, _ in enumerate(classified_ids):
    lc = lightcurve(classified_ids[ix])
    print(df_cat[df_cat.TransientID == classified_ids[ix]])
    plot_lightcurve(lc)
#    plt.savefig('graphs/binary/trans_corr_{}.png'.format(classified_ids[ix]))
#    plt.close()

missclassified_ids = [1509251350694128317, 1603021070274145695, 1404301350644109127, 1607060121174118737]
for ix, _ in enumerate(missclassified_ids):
    lc = lightcurve(missclassified_ids[ix])
    print(df_cat[df_cat.TransientID == missclassified_ids[ix]])
    plot_lightcurve(lc)
    plt.savefig('graphs/binary/trans_incorr_{}.png'.format(missclassified_ids[ix]))
    plt.close()

