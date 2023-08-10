get_ipython().system('mkdir -p data/downloads     && cd data/downloads     && wget -c "https://dl.dropboxusercontent.com/u/8680991/mafs/tcga_pancancer_dcc_mafs_082115.tar.gz"     && tar zxf "tcga_pancancer_dcc_mafs_082115.tar.gz"     && cd ../..')

import glob
import pandas as pd

folder = "data/downloads/mafs" ## Extracted archive folder that contains all the MAFs
df = pd.DataFrame()
for maf_file in glob.glob("{}/*.maf".format(folder)):
    mdf = pd.read_csv(maf_file, delimiter="\t", comment="#", low_memory=False)
    if "t_alt_count" in mdf.columns:  ## Not all MAFs have this. If not, skip the study MAF
        # example file name: `tcga_laml_from_dcc.maf`
        file_name = maf_file.split("/")[-1].split("_")[1].upper()
        # Also corresponds to the study abbreviation
        mdf['Study'] = file_name
        # Shorten the ID to make it compatible with the other data set (see below)
        mdf['Tumor_Sample_Barcode'] = ["-".join(s.split("-")[0:4]) for s in mdf['Tumor_Sample_Barcode']]
        mdf['read_count'] = mdf.t_alt_count + mdf.t_ref_count
        mdf['VAF'] = mdf.t_alt_count / mdf.read_count
        # Only save columns that might be of interest to us
        mdf = mdf[['Study', 'Tumor_Sample_Barcode', 'VAF']]
        df = df.append(mdf)
df.head()

purities = df.groupby(['Study', 'Tumor_Sample_Barcode']).median()
purities['MPurity'] = [min(v, 1) for v in (purities.VAF / 0.5)]  ## min -> safety check for right-skewed data
purities['Study'] = [i[0] for i in purities.index]
purities['Sample'] = [i[1] for i in purities.index]
purities.index = purities.Sample
purities.head()

import seaborn as sns

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
sns.set(color_codes=True)

sns.factorplot(x="Study", y="MPurity", hue="Study", data=purities, kind="box", aspect=4.5)
plt.show()

tcga = pd.read_csv("data/2-ncomms9971_s2.csv")  ## From the supplementary material
tcga.index = tcga["Sample ID"]
tcga.head()

# Plot the consensus purity estimate
sns.factorplot(x="Cancer type", y="CPE", hue="Cancer type", data=tcga, kind="box", aspect=4.5)
plt.show()

all_ests = pd.concat([tcga.drop(["Sample ID", "Cancer type"], axis=1), purities["MPurity"]], axis=1, join="inner")
# We want to work with complete data, so let's drop the patients that lack at least one of the estimates
all_ests = all_ests.dropna()
all_ests.head()

sns.pairplot(all_ests, dropna=True, kind='reg')

distance = lambda column1, column2: ((column1 - column2) ** 2).mean(axis=None)  ## MSE
mse_matrix = all_ests.apply(lambda col1: all_ests.apply(lambda col2: distance(col1, col2)))
sns.clustermap(mse_matrix)

