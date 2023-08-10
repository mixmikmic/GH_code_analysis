# Import for plotting
import matplotlib.pyplot as plt

# Get returns data for 5 different assets
assets = ['XLK', 'MIG', 'KO', 'ATHN', 'XLY']
data = get_pricing(assets,fields='price',start_date='2014-01-01',end_date='2015-01-01').pct_change()[1:].T

# Print pairwise correlations
print 'Correlation matrix:\n', np.corrcoef(data)

# Print the mean return of each
print 'Means:\n', data.T.mean()

# Plot what we've identified as the most and the least correlated pairs from the matrix above
plt.scatter(data.iloc[4], data.iloc[0], alpha=0.5)
plt.scatter(data.iloc[2], data.iloc[1], color='r', alpha=0.4)
plt.legend(['Correlated', 'Uncorrelated']);

