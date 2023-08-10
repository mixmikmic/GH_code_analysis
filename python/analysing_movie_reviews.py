import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set_style('white')

movies = pd.read_csv('fandango_score_comparison.csv')
dimensions = movies.shape
cols = movies.columns
print(cols)
print(dimensions)
plt.hist(movies['Fandango_Stars'])
plt.title('Fandango Stars')
plt.show()
plt.hist(movies['Metacritic_norm_round'])
plt.title('Metacritic norm round')
plt.show()

fan_mean = movies['Fandango_Stars'].mean()
met_mean = movies['Metacritic_norm_round'].mean()
fan_median = movies['Fandango_Stars'].median()
met_median = movies['Metacritic_norm_round'].median()
fan_std = movies['Fandango_Stars'].std()
met_std = movies['Metacritic_norm_round'].std()

print(fan_mean);print(fan_median);print(fan_std)
print(met_mean);print(met_median);print(met_std)

plt.scatter(movies['Metacritic_norm_round'], movies['Fandango_Stars'])
plt.xlabel('Metacritic')
plt.ylabel('Fandango')
plt.show()

movies['fm_diff'] = np.absolute(movies['Metacritic_norm_round'] - movies['Fandango_Stars'])
movies.sort('fm_diff', ascending=False).head()

from scipy.stats import pearsonr, linregress

r_value, p_value = pearsonr(movies['Metacritic_norm_round'], movies['Fandango_Stars'])

slope, intercept, r_value, p_value, stderror = linregress(movies['Metacritic_norm_round'], movies['Fandango_Stars'])

def predict_score(x_value):
    return slope * x_value + intercept
predict_score(3.0)

predict_score(4.0)

plt.scatter(movies['Metacritic_norm_round'], movies['Fandango_Stars'])
x = [i for i in range(0,6)]
y = [predict_score(i) for i in x]
plt.plot(x,y,color='red')
plt.show()



