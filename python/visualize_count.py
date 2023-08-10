import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

def myplot(df):
    xClm="count_sim"
    yClm="count_real"
    plt.figure(figsize=(8,8))
    sns.regplot(xClm,yClm,data=df,fit_reg=False)
    plt.plot([0,6],[0,6],"k:")
    plt.xlim([0,2.7])
    plt.ylim([0,2.7])
    plt.xlabel("")
    plt.ylabel("")
    plt.tick_params(labelsize=15)
    plt.show()

countFilepath="summarize_count.csv"
count_df=pd.read_csv(countFilepath)
print(count_df.shape)
count_df.head()

xClm="count_sim"
yClm="count_real"
g=sns.jointplot(xClm, yClm, data=count_df,stat_func=None)
g.ax_joint.plot(range(7), ':k')
g.ax_joint.set_xlim([0,2.7])
g.ax_joint.set_ylim([0,2.7])
g.ax_joint.set_xlabel("")
g.ax_joint.set_ylabel("")

#g.set(xlabel="",ylabel="")
plt.show()

myplot(count_df)

lookupFilepath="../speciespick/picked_assembly_summary_code.csv"
lookup_df=pd.read_csv(lookupFilepath)
print(lookup_df.shape)
lookup_df.head()

compFilepath="../preprocess/out/summary_composition.csv"
comp_df=pd.read_csv(compFilepath)
comp_df=comp_df[comp_df["dna_type"]=="chromosome"]#exclude plasmid
print(comp_df.shape)
comp_df.head()

tmp_df=pd.merge(count_df, lookup_df, on="taxid",how="left")
tmp_df=tmp_df[list(count_df.columns)+["organism_name","ftp_basename","domain","genetic_code"]]
tmp_df=tmp_df[tmp_df["genetic_code"]==11]#!!!TBI!!! failed to handle genetic code=4 for now
print(tmp_df.shape)
tmp_df.head()

myplot(tmp_df[tmp_df["domain"]=="archaea"])

out_df=pd.merge(tmp_df, comp_df, on="ftp_basename", how="left")
out_df=out_df[list(tmp_df.columns)+["G+C"]]
print(out_df.shape)
out_df.head()

out_df["diff"]=out_df["count_real"]-out_df["count_sim"]

plt.figure(figsize=(8,8))
sns.regplot("G+C","diff",data=out_df,fit_reg=False)
plt.show()

g=sns.jointplot("G+C", "diff", data=out_df,stat_func=None)
plt.show()

sns.distplot(out_df["G+C"], kde=False, bins =50)

out_df[(out_df["G+C"]<0.4) & (out_df["diff"]>0.2)]

out_df[(out_df["diff"]>0.37)]



