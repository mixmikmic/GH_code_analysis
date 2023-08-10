get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import sys
sys.path.append('/Users/kaonpark/workspace/github.com/likejazz/kaon-learn')
import kaonlearn
from kaonlearn.plots import plot_decision_regions

from sklearn.datasets import load_iris
iris = load_iris()

iris.data[:5]

classes = np.array(["sentosa", "versicolor", "virginica"])

iris_data = np.array(iris.data)
iris_data = pd.DataFrame(iris_data)
iris_data['species'] = pd.Series(
    classes[iris.target],
    index=iris_data.index, dtype=str)

iris_data = iris_data.rename(columns={
    0: 'sepal_length', 
    1: 'sepal_width', 
    2: 'petal_length', 
    3: 'petal_width'})

iris_data.head()

sns.pairplot(iris_data, hue="species")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)
iris.data[:3], X_scaled[:3]

from sklearn.decomposition import PCA
# keep the first two principal components of the data
pca = PCA(n_components=2)
# fit PCA model to beast cancer data
pca.fit(X_scaled)

# transform data onto the first two principal components
X_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
p = make_pipeline(SVC(C=1))
p.fit(X_pca, iris.target)

# plot fist vs second principal component, color by class
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.title("SVC to classify a dataset with applying PCA")
plot_decision_regions(X_pca, iris.target, p, target_names=iris.target_names, shows=True)

from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)

X_tsne = tsne.fit_transform(iris.data)

p = make_pipeline(SVC(C=1, gamma=0.001))
p.fit(X_tsne, iris.target)

plt.xlabel("t-SNE feature 0")
plt.ylabel("t-SNE feature 1")
plt.title("SVC to classify a dataset with applying t-SNE")
plot_decision_regions(X_tsne, iris.target, p, target_names=iris.target_names, shows=True)

X_tsne = tsne.fit_transform(X_scaled)

p = make_pipeline(SVC(C=1, gamma=0.001))
p.fit(X_tsne, iris.target)

plt.xlabel("t-SNE feature 0")
plt.ylabel("t-SNE feature 1")
plt.title("SVC to classify a dataset with applying t-SNE, StandardScaler")
plot_decision_regions(X_tsne, iris.target, p, target_names=iris.target_names, shows=True)

import pandas as pd
# create a dataframe with an integer feature and a categorical string feature
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                        'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
demo_df

pd.get_dummies(demo_df)

demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature'])

kaonlearn.plots.plot_scaling()

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
cancer.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cancer.data)
X_scaled

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

clf = SVC(C=1, gamma=0.01).fit(X_pca, cancer.target)

plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.title("SVC, PCA, StandardScaler")
plot_decision_regions(X_pca, cancer.target, clf=clf, res=1, target_names=cancer.target_names, shows=True)

pca.components_

pca.components_.shape

pca_data = pd.DataFrame(data=pca.components_, columns=cancer.feature_names)

plt.figure(figsize=(30,2))
sns.heatmap(pca_data, yticklabels=["First component", "Second component"], annot=True)

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(cancer.data)

clf = SVC(C=1, gamma=0.1).fit(X_tsne, cancer.target)

plt.title("SVC, t-SNE")
plot_decision_regions(X_tsne, cancer.target, clf=clf, res=1, target_names=cancer.target_names, shows=True)

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(scaler.fit_transform(cancer.data))

clf = SVC(C=1, gamma=0.1).fit(X_tsne, cancer.target)

plt.title("SVC, t-SNE, StandardScaler")
plot_decision_regions(X_tsne, cancer.target, clf=clf, res=1, target_names=cancer.target_names, shows=True)

