# allow import without install
import sys
if ".." not in sys.path:
    sys.path.append("..")

import nearest_neighbors
from sklearn import neighbors, datasets

# This visualization illustrates the correlations among the features and the species distributions
# for feature pairs.
reload(nearest_neighbors)
iris = datasets.load_iris()
#bc = datasets.load_breast_cancer()
w = nearest_neighbors.SVGScatterer(iris).widget()

# This widget interactively compares two features and the nearest neighbor boundary
# implied by using just those features to species.
nearest_neighbors.load_javascript_support()
nearest_neighbors.show_iris()

# This visualization extends the nearest neighbor visualization to 3 dimension.
import nn3d

nn = nn3d.show_iris()

nn.x_name, nn.y_name, nn.z_name

nn.scatter_plot.results

from jp_gene_viz import js_context


js_context.load_if_not_loaded(["three.js"], force=True, verbose=True)



