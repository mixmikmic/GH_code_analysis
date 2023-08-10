import pandas as pd
import numpy as np
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic('matplotlib inline')

# Load the dataset
try:
    data = pd.read_csv("SanFranSalary.csv", index_col='Id')
    data = data[data.Year == 2014]
    print "Salary dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

data.dtypes

data.isnull().sum()

data.tail()

data = data[data.JobTitle != 'Not Provided'] # Remove records where JobTitle is `Not Provided`

data = data[data.BasePay != 'Not Provided'] # Remove records where BasePay is `Not Provided`

data['Benefits'].fillna(0.00, inplace=True) # Fill dirty `Benefits` records with `0.00`

del data['EmployeeName']

del data['Notes']

del data['Agency']

del data['Status']

data.dropna(inplace=True)

job_title = data['JobTitle']

year = data['Year']

del data['JobTitle']

del data['Year']

data['OvertimePay'].replace('Not Provided', 0.00)

data['OtherPay'].replace('Not Provided', 0.00)

data['Benefits'].replace('Not Provided', 0.00)

data = data.apply(pd.to_numeric) # Force dtype conversion

# Remove negative entries from the dataset
data = data[data.BasePay > 0]
data = data[data.OvertimePay > 0]
data = data[data.OtherPay > 0]
data = data[data.Benefits > 0]

print 'New shape of the data is ({},{})'.format(*data.shape)

print data.isnull().sum()

data.tail()

data.describe()

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

scaled_data = np.log(data)

pd.scatter_matrix(scaled_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# Q1 (25th percentile of the data) for the 'TotalPayBenefits'
Q1 = np.percentile(scaled_data['TotalPayBenefits'], 25)
    
# Q3 (75th percentile of the data) for the 'TotalPayBenefits'
Q3 = np.percentile(scaled_data['TotalPayBenefits'], 75)
    
# Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
step = 1.5*(Q3-Q1)
    
# Display the outliers
print "Data points considered outliers for the feature '{}':".format('TotalPayBenefits')
display(scaled_data[~((scaled_data['TotalPayBenefits'] >= Q1 - step) & (scaled_data['TotalPayBenefits'] <= Q3 + step))])

# Remove the outliers
scaled_and_trimmed_data = scaled_data[((scaled_data['TotalPayBenefits'] >= Q1 - step) & (scaled_data['TotalPayBenefits'] <= Q3 + step))]

print 'Final shape of the dataset is ({}, {})'.format(*scaled_and_trimmed_data.shape)

pd.scatter_matrix(scaled_and_trimmed_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

from sklearn.decomposition import PCA
# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=6)
pca.fit(scaled_and_trimmed_data)

# Generate PCA results plot
pca_results = vs.pca_results(scaled_and_trimmed_data, pca)

pca = PCA(n_components=2)
pca.fit(scaled_and_trimmed_data)
# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(scaled_and_trimmed_data)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Create a biplot
vs.biplot(scaled_and_trimmed_data, reduced_data, pca)

from sklearn.mixture import GMM
from sklearn.metrics import silhouette_score

clusters = 3

clusterer = GMM(n_components=clusters, random_state=42)

clusterer.fit(reduced_data)

preds = clusterer.predict(reduced_data)

centers = clusterer.means_

score = silhouette_score(reduced_data, preds)

print "Silhouette Score for {} clusters is {}".format(clusters, score)
    
vs.cluster_results(reduced_data, preds, centers)

log_centers = pca.inverse_transform(centers)


true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

cluster_0 = []
cluster_1 = []
cluster_2 = []

predictions = pd.DataFrame(preds, columns = ['Cluster'])
plot_data = pd.concat([predictions, reduced_data], axis = 1)
        
cluster_0 = plot_data[plot_data.Cluster == 0]
cluster_1 = plot_data[plot_data.Cluster == 1]
cluster_2 = plot_data[plot_data.Cluster == 2]

del cluster_0['Cluster']
del cluster_1['Cluster']
del cluster_2['Cluster']

cluster_0 = np.exp(pca.inverse_transform(cluster_0))
cluster_0 = pd.DataFrame(np.round(cluster_0), columns = scaled_and_trimmed_data.keys())
cluster_1 = np.exp(pca.inverse_transform(cluster_1))
cluster_1 = pd.DataFrame(np.round(cluster_1), columns = scaled_and_trimmed_data.keys())
cluster_2 = np.exp(pca.inverse_transform(cluster_2))
cluster_2 = pd.DataFrame(np.round(cluster_2), columns = scaled_and_trimmed_data.keys())

print 'Cluster - 0 Stats'
display(cluster_0.describe())
print 'Cluster - 1 Stats'
display(cluster_1.describe())
print 'Cluster - 2 Stats'
display(cluster_2.describe())



