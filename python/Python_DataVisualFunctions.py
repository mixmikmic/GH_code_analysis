## Example to read file and create dataframe

import pandas as pd
from matplotlib import pyplot as plt
import sklearn.datasets


def get_iris_df():
    ds = sklearn.datasets.load_iris()
    df = pd.DataFrame(ds['data'], columns=ds['feature_names'])
    code_species_map = dict(zip(range(3), ds['target_names']))
    df['species'] = [code_species_map[c] for c in ds['target']]
    return df
df = get_iris_df()
df.head

## Pie chart

sums_by_species = df.groupby('species').sum()
var = 'sepal width (cm)'
sums_by_species[var].plot(kind='pie', fontsize=20)
plt.ylabel(var, horizontalalignment='left')
plt.title('Breakdown for ' + var, fontsize=25)
#plt.savefig('iris_pie_for_one_variable.jpg')
plt.show()
plt.close()

## multiple charts on each coloumn

sums_by_species = df.groupby('species').sum()
sums_by_species.plot(kind='pie', subplots=True,
layout=(2,2), legend=False)
plt.title('Total Measurements, by Species')
#plt.savefig('iris_pie_for_each_variable.jpg')
plt.show()
plt.close()

## Bar chart

sums_by_species = df.groupby('species').sum()
var = 'sepal width (cm)'
sums_by_species[var].plot(kind='bar', fontsize=15,
rot=30)
plt.title('Breakdown for ' + var, fontsize=20)
#plt.savefig('iris_bar_for_one_variable.jpg')
plt.show()
plt.close()
sums_by_species = df.groupby('species').sum()
sums_by_species.plot(
kind='bar', subplots=True, fontsize=12)
plt.suptitle('Total Measurements, by Species')
#plt.savefig('iris_bar_for_each_variable.jpg')
plt.show()
plt.close()

## Histograms

df.plot(kind='hist', subplots=True, layout=(2,2))
plt.suptitle('Iris Histograms', fontsize=20)
plt.show()

## Histogram on same axes

for spec in df['species'].unique():
    forspec = df[df['species']==spec]
    forspec['petal length (cm)'].plot(kind='hist', alpha=0.4, label=spec)
    plt.legend(loc='upper right')
    plt.suptitle('Petal Length by Species')
    #plt.savefig('iris_hist_by_spec.jpg')
plt.show()

# Means, Standard Deviations, Medians and Quartiles

col = df['petal length (cm)']
Average = col.mean()
print ("Average length of petal ", Average)
Std = col.std()
print ("Standard deviation ", Std)
Median = col.quantile(0.5)
print ("Median of petals are ", Median)
Percentile25 = col.quantile(0.25)
print ("Quantiles wth percentile 25 ", Percentile25)
Percentile75 = col.quantile(0.75)
print ("Quantiles wth percentile 75 ", Percentile75)

## REmove outliers from data before calculating avg

col = df['petal length (cm)']
Perc25 = col.quantile(0.25)
Perc75 = col.quantile(0.75)
Clean_Avg = col[(col>Perc25)&(col<Perc75)].mean()
print ("Clean Average ", Clean_Avg)

## Box plot of sepal

col = 'sepal length (cm)'
df['ind'] = pd.Series(df.index).apply(lambda i: i% 50)
df.pivot('ind','species')[col].plot(kind='box')
plt.show()

## Scatter plot on the sepal length and width

df.plot(kind="scatter",
x="sepal length (cm)", y="sepal width (cm)")
plt.title("Length vs Width")
plt.show()
plt.close()

## Scatter plot with colors 

colors = ["r", "g", "b"]
markers= [".", "*", "^"]
fig, ax = plt.subplots(1, 1)
for i, spec in enumerate(df['species'].unique()):
    ddf = df[df['species']==spec]
    ddf.plot(kind="scatter",x="sepal width (cm)", y="sepal length (cm)",
    alpha=0.5, s=10*(i+1), ax=ax,
    color=colors[i], marker=markers[i], label=spec)
plt.legend()
plt.show()

## Scatter plot with the logarithmic axis

import pandas as pd
import sklearn.datasets as ds
import matplotlib.pyplot as plt
# Make Pandas dataframe
bs = ds.load_boston()
df = pd.DataFrame(bs.data, columns=bs.feature_names)
df['MEDV'] = bs.target
# Normal Scatterplot
df.plot(x='CRIM',y='MEDV',kind='scatter')
plt.title('Crime rate on normal axis')
plt.show()

print ("---------")
print ("After changing the axis with the log axis")

df.plot(x='CRIM',y='MEDV',kind='scatter',logx=True)
plt.title('Crime rate on logarithmic axis')
plt.show()
plt.close()

## Scatter Matrices


from pandas.plotting import scatter_matrix
scatter_matrix(df)
plt.show()
plt.close()

## Heatmaps

df.head

df.plot(kind='hexbin', x='sepal width (cm)', y='sepal length (cm)')
plt.show()
plt.close()

## Correlation

# Pearson corr
print (df["sepal width (cm)"].corr(df["sepal length (cm)"]))
print (df["sepal width (cm)"].corr( df["sepal length (cm)"], method="pearson"))
print (df["sepal width (cm)"].corr( df["sepal length (cm)"], method="spearman"))
print (df["sepal width (cm)"].corr( df["sepal length (cm)"], method="spearman"))

## Simple example of time series data plot

import statsmodels.api as sm
dta = sm.datasets.co2.load_pandas().data
dta.plot()
plt.title("CO2 Levels")
plt.ylabel("Parts per million")
plt.show()

# Google stock since 2010 on a normal logarithmic axes
# Google stock data downloaded from https://finance.yahoo.com/quote/GOOG/history?p=GOOG&c=2000 

import urllib.request as request
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Get raw CSV data from the web

# Make DataFrame, w timestamp as the index
df = pd.read_csv('data_files/GOOG.csv')
df.index = df['Date'].astype('datetime64')
df['LogClose'] = np.log(df['Close'])
df['Close'].plot()
plt.title("Normal Axis")
plt.show()
df['Close'].plot(logy=True)
plt.title("Logarithmic Axis")
plt.show()

