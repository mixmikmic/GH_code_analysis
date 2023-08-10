import dask.dataframe as dd
import dask.array as da
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas
pdf = pd.read_csv('../data/ign.csv').drop('Unnamed: 0', axis=1)

# Dask
ddf = dd.read_csv('../data/ign.csv').drop('Unnamed: 0', axis=1).repartition(npartitions=5)

pdf

ddf

ddf.head()

# Pandas
pdf.isnull().sum()

# Dask
missing_vals = ddf.isnull().sum()
missing_vals

missing_vals.compute()

missing_vals.visualize()

# Pandas
p_genre_counts = pdf.genre.value_counts()
p_popular_genre_counts = p_genre_counts[p_genre_counts > 250]
p_popular_genre_counts.plot(kind='pie', autopct='%1.1f%%')
plt.show()

# Dask
d_genre_counts = ddf.genre.value_counts()

# Familiar filtering construct
d_popular_genre_counts = d_genre_counts[d_genre_counts > 250]

# Since DataFrame.compute() always returns a Pandas Series or DataFrame,
# we can use its built in PyPlot methods
d_popular_genre_counts.compute().plot(kind='pie', autopct='%1.1f%%')
plt.show()

d_popular_genre_counts.visualize()

# Pandas
bins = [0,1,2,3,4,5,6,7,8,9,10,11]
pdf.score.hist(bins=bins)
plt.show()

# Dask
# dask.dataframe does not have a native histogram method
# Manually binning in a dataframe can also be unwieldy (no cut method)
# Solution: Convert the Series to a dask.array
d_scores = ddf.score.values
h, bins = da.histogram(d_scores, bins=bins)
plt.bar(bins[:-1], h.compute())
plt.xlim(0,11)
plt.show()

# Pandas
plt.subplots(figsize=(15,15))

# Filter games to only the popular genres
p_games_in_popular_genres = pdf[pdf.genre.isin(p_popular_genre_counts.index)]

# Calculate the mean score by genre, by year
p_mean_score_by_year = p_games_in_popular_genres.groupby(['release_year','genre']).score.mean()

# Unstack and plot as a heatmap
sns.heatmap(p_mean_score_by_year.unstack(), annot=True, cmap='RdYlGn', linewidths=0.4, vmin=0, vmax=10)
plt.show()

# Dask
plt.subplots(figsize=(15,15))

# Filter games to only the popular genres
d_games_in_popular_genres = ddf[ddf.genre.isin(d_popular_genre_counts.index.compute())]

# Calculate the mean score by genre, by year
d_mean_score_by_year = d_games_in_popular_genres.groupby(['release_year','genre']).score.mean()

# Unstack and plot as a heatmap (unstack is not implemented in Dask; we collect the result then let Pandas reshape)
sns.heatmap(d_mean_score_by_year.compute().unstack(), annot=True, cmap='RdYlGn', linewidths=0.4, vmin=0, vmax=10)
plt.show()



