import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from geopy import Nominatim
import geojson
import folium
from branca.colormap import LinearColormap, StepColormap

get_ipython().run_line_magic('matplotlib', 'inline')

df_dirty = pd.read_csv('./data/sf/data.csv')
df_dirty.head(5) # display first 5 entries of DataFrame

# globally set our seaborn plot size to 12 by 8 inches:
sns.set(rc={'figure.figsize':(12, 8)})

def plot_prices(dataframe: pd.DataFrame, bins: list):
    fig, ax = plt.subplots()
    ax.set_xticks(bins)
    plt.xticks(rotation='vertical')
    return sns.distplot(dataframe.price, bins=bins)

bins = range(int(df_dirty.price.min()),int(df_dirty.price.max()),500000)
bins
plot_prices(df_dirty.dropna(), bins)
print(f'Skewness: {df_dirty.price.skew()}')
print(f'Kurtosis: {df_dirty.price.kurt()}')

print(f'max price before: {df_clean.price.max()}')
cutoff = 8e6
df_clean = df_dirty[df_dirty['price'] <= cutoff]
print(f'max price after: {df_clean.price.max()}')

bins = range(int(df_clean.price.min()),int(df_clean.price.max()),500000)
plot_prices(df_clean, bins)
print("Skewness: %f" % df_clean['price'].skew())
print("Kurtosis: %f" % df_clean['price'].kurt())

num_zero_sqft = (df_clean['sqft'] < 10).sum()
print("There are {} entries with zero sqft".format(num_zero_sqft))

df_clean = df_clean[df_clean['sqft'] > 10]
num_zero_sqft = (df_clean['sqft'] < 10).sum()
print("There are {} entries with zero sqft".format(num_zero_sqft))

sns.regplot(df_clean['sqft'], df_clean['price'], fit_reg=False)

print(f'max sqft before: {df_clean.sqft.max()}')
df_clean = df_clean[df_clean['sqft'] < 9000]
print(f'max sqft after: {df_clean.sqft.max()}')

df_clean.info()

missing = df_clean.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
plt.title("Counts of Missing Values")
plt.show()
missing_ratio = missing / len(df_clean)
missing_ratio.plot.bar()
plt.title("Ratio of Missing Values")
plt.show()

print(df_clean.columns)
df_clean = df_clean.drop(columns=['latlng', 'real estate provider'])
print(df_clean.columns)

df_clean_dropna = df_clean.dropna()

from sklearn.preprocessing import Imputer
df_clean_imputed = df_clean.copy() # copy original for safe keeping
columns_to_impute = ['bed', 'bath', 'sqft'] # only impute numerical columns
imputer = Imputer(strategy='mean')
imputed_columns = imputer.fit_transform(df_clean_imputed[columns_to_impute])
df_clean_imputed[columns_to_impute] = imputed_columns
df_clean_imputed.info()

df_clean_imputed = df_clean_imputed.dropna()

df_clean_imputed = df_clean_imputed[df_clean_imputed.postal_code != 94501] 

df_clean_imputed.info()

df_clean_dropna.to_csv('./data/sf/data_clean_dropna.csv', index=False)
df_clean_imputed.to_csv('./data/sf/data_clean_imputed.csv', index=False)

