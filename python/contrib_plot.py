import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

filepath="/Users/mitsuki/rsyncdir/files/out.csv"
df=pd.read_csv(filepath)
df.head()

columns_lst=[]
for i in range(16):
    for j in range(16):
        columns_lst.append("{}->{}".format(i,j))
df[columns_lst].sum(axis=1).head()

plt.figure(figsize=(10,10))
sns.regplot("G+C","count_diff",data=df,fit_reg=False)
plt.show()

df[columns_lst].describe()

patternDegs_lst=['000000',
                                '100000',
                                '110000',
                                '100100','100010','100001',
                                '111000',
                                '110100','110010','110001',
                                '111100',
                                '110110','110011','110101',
                                '111110',
                                '111111']

patternDegs_lst[15]

target_lst=[]
for i in range(16):
    for j in range(16):
        clm="{0}->{1}".format(i,j)
        thres=0.05
        overThresCount=np.sum(df[clm]>thres)
        if overThresCount>0:
            print("{0}->{1}: {2}".format(patternDegs_lst[i], patternDegs_lst[j], overThresCount))
            target_lst.append((i,j))
print(target_lst)

plt.figure(figsize=(10,10))
sns.regplot("G+C","count_diff",data=df,fit_reg=False, label="total")

for i,j in target_lst:
    clm="{0}->{1}".format(i,j)
    label="{0}->{1}".format(patternDegs_lst[i], patternDegs_lst[j])
    sns.regplot("G+C",clm,data=df,fit_reg=False, label=label)
    #sns.regplot("G+C",clm,data=df,fit_reg=False, label=clm)
        
plt.legend()
plt.show()

df.head()

df["remain"]=df["count_diff"]-(df["3->1"]+df["8->1"]+df["8->3"])

plt.figure(figsize=(10,10))
sns.regplot("G+C","count_diff",data=df,fit_reg=False, label="total")
sns.regplot("G+C","remain",data=df,fit_reg=False, label="remain")



for key, val in  (df.loc[np.argmax(df["remain"]),columns_lst]).iteritems():
    thres=0.005
    if val>thres:
        i,j=[int(_) for _ in  key.split("->")]
        label="{0}->{1}".format(patternDegs_lst[i], patternDegs_lst[j])
        print("{0} ({1}) : {2}".format(label, key, val))

