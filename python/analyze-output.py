import pandas as pd
import matplotlib

get_ipython().magic('matplotlib inline')

results = pd.read_csv('build/eval-results.csv')
results.head()

agg_results = results.drop(['Partition'], axis=1).groupby('Algorithm').mean()
agg_results

results.loc[:,['Algorithm', 'RMSE.ByUser']].boxplot(by='Algorithm')

results.loc[:,['Algorithm', 'Predict.nDCG']].boxplot(by='Algorithm')

results.loc[:,['Algorithm', 'BuildTime', 'TestTime']].boxplot(by='Algorithm')



