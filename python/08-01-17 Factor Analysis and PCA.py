import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import FactorAnalysis

iris = load_iris()
X,y = iris.data, iris.target
factor = FactorAnalysis(n_components=4, random_state=101).fit(X)

pd.DataFrame(factor.components_ , columns=iris.feature_names)

import pandas as pd
from sklearn.decomposition import PCA

pca = PCA().fit(X)
print ('Explained variance by component: %s' %pca.explained_variance_ratio_)

pd.DataFrame(pca.components_, columns = iris.feature_names)



