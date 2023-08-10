# allow import without install
import sys
if ".." not in sys.path:
    sys.path.append("..")

# Load a pair of networks with coordinated display.
from jp_gene_viz import paired_networks
paired_networks.load_javascript_support()

P = paired_networks.PairedNetworks()
P.load_networks("network.tsv", "network2.tsv")

P.show()

from jp_gene_viz import paired_links
LL = paired_links.PairedLinks()
LL.load_left("network.tsv", "expr.tsv")
LL.load_right("network2.tsv", "expr.tsv")

LL.show()



