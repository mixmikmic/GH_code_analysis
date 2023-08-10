# Uncomment to run without install in the Binder service.
import sys
if ".." not in sys.path:
    sys.path.append("..")

# Import the python implementation.
from jp_gene_viz import dNetwork

# Load the javascript "client side" support logic.
dNetwork.load_javascript_support()

# Load and display the network.
N = dNetwork.display_network("network.tsv")
N.title_html.value = "An example regulatory network"

# randomly reset the weight to integers 0..10

def adjust_colorization():
    import random
    G = N.display_graph
    nw = G.node_weights
    for node in nw.keys():
        nw[node] = random.randint(0, 10)
    ew = G.edge_weights
    for edge in ew.keys():
        ew[edge] = random.randint(0,10)
    # Use Emily's color scheme, please refer to the source code for more information.
    dNetwork.set_node_color_levels(N)
    dNetwork.set_edge_color_levels(N)
    # redraw
    N.draw()

# Uncomment to change the network colorization
#adjust_colorization()



