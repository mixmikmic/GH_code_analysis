# allow import without install
import sys
if ".." not in sys.path:
    sys.path.append("..")
import nearest_neighbors
from sklearn import neighbors, datasets

# This visualization illustrates the correlations among the features and the malign/benign distributions
# for feature pairs.

bc = datasets.load_breast_cancer()
w = nearest_neighbors.SVGScatterer(bc).widget()

# This widget interactively compares two features and the nearest neighbor boundary
# implied by using just those features to classify benign versus malignant tumors.
nearest_neighbors.load_javascript_support()
nearest_neighbors.show_bc()

# This visualization extends the nearest neighbor visualization to 3 dimension.
import nn3d

nn = nn3d.show_bc()

nn.x_name, nn.y_name, nn.z_name

nn.scatter_plot.results

from jp_gene_viz import js_context


js_context.load_if_not_loaded(["three.js"], force=True, verbose=True)



