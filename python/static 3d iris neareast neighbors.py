# allow import without install
import sys
if ".." not in sys.path:
    sys.path.append("..")

# This visualization extends the nearest neighbor visualization to 3 dimension.
import nn3d

nn = nn3d.static_iris()

print nn.scatter_plot.embedded_html()



