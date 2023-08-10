import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

df = pd.read_csv('amino_acid_genotypes_to_brightness.tsv', sep = '\t', engine = 'python')
df.head()

mutants = df['aaMutations']       #split to list of aminoacids
mutants = mutants.str.split(':')
mutants = mutants[1:]
mutants.head()

pd.Series(mutants[2])

mut_separated = pd.Series(mutants[2]).str.extract('(?P<position>^[A-Z]{2}\d+)(?P<mutation>[A-Z]$)')

mut_separated

mut_location = []
for i in xrange(1,len(mutants)):
    mut_location.append(pd.Series(mutants[i]).str.extract('(^[A-Z]{2}\d+)'))

df_mut_location = pd.DataFrame(mut_location)
df_mut_location.head()

boleans_test = ~df_mut_location.isnull()
boleans_test_hist = boleans_test.sum(axis = 1)

plt.figure()
plt.hist(boleans_test_hist, bins=15)
plt.show()

uniq_mut_los=np.unique(df_mut_location.values.ravel()) # unique values of locations

uniq_mut_los = uniq_mut_los[1:] #droping the NaN entry 
len(uniq_mut_los)

cols = uniq_mut_los.tolist()  #unique values of locations -> new columns

len(cols)

categ_mat = np.zeros((len (df_mut_location), len (cols)))

for i in xrange(len(df_mut_location)):
    for j in xrange(len(cols)):
        if cols[j] in df_mut_location.values[i]:
            categ_mat[i,j]=1
        else:
            categ_mat[i,j]=0
            

categ_mat.shape

plt.figure()
plt.hist(categ_mat.sum(axis=1), bins=15)
plt.show()

all(boleans_test_hist == categ_mat.sum(axis=1)) #out: TRUE!

locations_df = pd.DataFrame(categ_mat, columns=cols)

locations_df.head()

brightness_df = df.ix[1:]

brightness_df = brightness_df.reset_index()
brightness_df = brightness_df.drop(['index'], axis=1)

brightness_df.head()

locations_df = pd.concat((locations_df,brightness_df),axis=1)

locations_df = locations_df.drop(['aaMutations'], axis = 1)

locations_df.tail()



