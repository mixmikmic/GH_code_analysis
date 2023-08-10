import seaborn as sns
import pandas as pd

# This is a commnet. The line the follows allows for inline plotting
get_ipython().magic('matplotlib inline')

df = sns.load_dataset("anscombe")

# What is the command doing?
df.head(5)

df.shape

df.max()

#df.min()

#df.mean()

# There are four datasets in the dataframe. I, II, III, IV
# How do we get the mean for each one
df1 = df.loc[df['dataset'] == 'I']

df1.mean()

# What is this doing?
df1.cov()

# What about this?
df1.corr()





# What are the means of all the datasets?
# MEAN:

# When you remove a "#" this allows the command to be run in the notebook. 

#sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
#           col_wrap=2, ci=None, palette="muted", size=4,
#           scatter_kws={"s": 50, "alpha": 1})

bat_df = pd.read_csv('bat_data.txt',sep="\t")

bat_df.head(5)

g = sns.jointplot("lat", "vsd_richness", kind="reg", data=bat_df)
# There are other kind="" plots to make. Try the following: hex and kde

# Copy and paste line 71 down here. Try and create a plot that shows Actinobacteria counts plotted against latitude


# g = sns.jointplot("ecoregion_iv", "Actinobacteria", kind="reg", data=bat_df)
# Uncomment the line above and run it. 

sns.boxplot(x="ecoregion_iv", y="Actinobacteria", data=bat_df)

sns.boxplot(x="ecoregion_iv", y="Actinobacteria", data=bat_df)
sns.despine()

plot3 = sns.boxplot(x="ecoregion_iv", y="Actinobacteria", data=bat_df)
sns.despine()
for item in plot3.get_xticklabels():
    item.set_rotation(90)

plot4 = sns.boxplot(x="ecoregion_iv", y="Actinobacteria", data=bat_df)
sns.despine(left=True,bottom=True)
for item in plot4.get_xticklabels():
    item.set_rotation(90)

plot5 = sns.boxplot(x="ecoregion_iv", y="Actinobacteria", data=bat_df)
sns.despine(left=True,bottom=True)
for item in plot5.get_xticklabels():
    item.set_rotation(90)

plot6 = sns.violinplot(x="ecoregion_iv", y="Actinobacteria", data=bat_df)
sns.despine()
for item in plot6.get_xticklabels():
    item.set_rotation(90)



