# Numpy
import numpy as np
from numpy import concatenate, array
from numpy.random import randn
# Decimal precision value to display in the matrix
np.set_printoptions(precision=5, suppress=True)

# Scipy
import scipy
import scipy.stats as stats

# Matplotlib
import matplotlib.pyplot as pyplot
get_ipython().magic('matplotlib inline')

# Pandas experiments
import pandas as pd
from pandas import Series, DataFrame, Panel

# Misc
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
print 'All libraries loaded.'

# File path
combined_file_path = '~/code/independent/datasets/combined_df_removedcolumns.csv'

def calculateSimilarity(col1_name, col2_name, df):
    smalldf = df[[col1_name, col2_name]]
    smalldf = smalldf.dropna(axis=0)

#     rho = pearsonr(smalldf[col1_name].values, smalldf[col2_name].values)[0]
    rho = spearmanr(smalldf[col1_name].values, smalldf[col2_name].values)[0]
    return rho

corr=[]
findings_df = pd.read_csv(combined_file_path, delimiter='\t')
gfr=list(findings_df.columns.values)[-1]
col_list=list(findings_df.columns.values)[6:-1]
for ele in col_list:
    corr.append(calculateSimilarity(gfr,ele,findings_df))

# Plot the correlation results
figure = pyplot.figure()
figure.set_size_inches(16.5, 4.5, forward=True)
indices = [i for i,val in enumerate(corr)]
pyplot.plot(indices, corr, marker='o', linewidth=2.5)
pyplot.grid()
pyplot.xticks(indices, col_list, rotation=90)
pyplot.show()

