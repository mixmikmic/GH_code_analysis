import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

df = pd.read_csv('amino_acid_genotypes_to_brightness.tsv', sep = '\t', engine = 'python')

#split to list of aminoacids
mutants = df.aaMutations.str.split(':')
mutants = mutants[1:]
mutants.head()

mta_dic = mutants.to_dict()
mutation_df = df = pd.DataFrame.from_dict(mta_dic, orient='index')
mutation_df.head()

uniq_mut=np.unique(mutation_df.values.ravel()) # unique values of locations

uniq_mut= uniq_mut[1:] #droping the NaN entry 
len(uniq_mut)

uniq_mut

cols = uniq_mut.tolist()

len(cols)



# # # Danger
categ_mat = np.zeros((len (mutation_df), len (cols)))

for i in xrange(len(mutation_df)):
    for j in xrange(len(cols)):
        if cols[j] in mutation_df.values[i]:
            categ_mat[i,j]=1
        else:
            categ_mat[i,j]=0
            

categ_mat.shape

plt.figure()
plt.hist(categ_mat.sum(axis=1), bins=15)
plt.show()

mutations_categ_df = pd.DataFrame(categ_mat, columns=cols)

mutations_categ_df.head()

mutation_categ_int_df = mutations_categ_df.astype(int)

mutation_categ_int_df.head()

df = df.ix[1:]
df = df.reset_index()
df = df.drop(['index'], axis=1)

df.head()

final_df = pd.concat((mutation_categ_int_df,df),axis=1)

final_df = final_df.drop(['aaMutations'], axis = 1)

final_df.head()

final_df.to_csv('mutation_based_df.csv')

