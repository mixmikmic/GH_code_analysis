

# Here is an example live usage of the widget:

# Uncomment to run without install.
import sys
if ".." not in sys.path:
    sys.path.append("..")
from jp_gene_viz import dNetwork
dNetwork.load_javascript_support()
N = dNetwork.display_network("../examples/network.tsv")



