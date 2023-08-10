# Uncomment to run without install (in binder, for example)
import sys
if ".." not in sys.path:
    sys.path.append("..")

from jp_gene_viz import dNetwork
dNetwork.load_javascript_support()

N1 = dNetwork.display_network("network.tsv", show=False)
N2 = dNetwork.display_network("network2.tsv", show=False)
N3 = dNetwork.display_network("network.tsv", show=False)

from jp_gene_viz import multiple_network

M = multiple_network.MultipleNetworks(
        [[N1, N2], 
         [N3]])

M.show()

# Configure the widgets after they have been fully displayed
N1.title_html.value = "First network"
N2.title_html.value = "Second network"
N2.threshhold_slider.value = 0.4
N2.apply_click(None)
N3.title_html.value = "just a copy of first network"

from jp_gene_viz import LExpression

# Create a linked network and heatmap.  Load the data.
L = LExpression.LinkedExpressionNetwork()
L.load_network("network.tsv")
L.load_heatmap("expr.tsv")

# Create a non-linked network.
N = dNetwork.display_network("network2.tsv", show=False)

# Create a combined widget, positioning the linked network below.
M2 = multiple_network.MultipleNetworks(
        [[N], 
         [L]])
M2.svg_width = 500

# Display the combined widget.
M2.show()

# Customize the component widgets after displaying them.
L.network.title_html.value = "Linked network (click checkbox to show heatmap)"
L.network.threshhold_slider.value = 6.7
L.network.apply_click(None)
L.load_heatmap("expr.tsv")

N.title_html.value = "Non-linked network"
N.threshhold_slider.value = 0.339
N.apply_click(None)



