directory = "/Users/emiraldi/erm/MariaP/Inferelator/input/GeneralPriors/Tr1"

networkInits = [("Tr1Irf1_comb_koATAC_sp.tsv","Tr1: Irf1KO v Control",'Tr1 Irf1 Prior'),
    ("Tr1Batf_comb_koATAC_sp.tsv","Tr1: BatfKO v Control",'Tr1 BatfKO Prior')]

baseCompNet = ("Tr1Prior_KO_ATAC_all_sp.tsv","72h Tr1:Th0 micro","Tr1 ATAC + KO Prior")

expressionFile = "Tr1_ThVals.txt"

threshhold = 4

baseNetFile = baseCompNet[0]
baseCol = baseCompNet[1].lower()
baseNetName = baseCompNet[2]

# allow import without install
import sys
if ".." not in sys.path:
    sys.path.append("..")

# Load a pair of networks with coordinated display.
from jp_gene_viz import paired_networks
paired_networks.load_javascript_support()
from jp_gene_viz import paired_links

for networkInit in networkInits:
    currCol = networkInit[1].lower()
    networkFile = networkInit[0]
    LL = paired_links.PairedLinks()
    LL.load_left(directory + '/' + baseNetFile, directory + '/' + expressionFile)
    LL.load_right(directory + '/' + networkFile, directory + '/' + expressionFile)
    LL.show()    
    
#     N = L.network
#     N.set_title(networkInit[2])
#     N.threshhold_slider.value = threshhold
#     N.apply_click(None)
#     N.draw()
#     L.gene_click(None)    
#     L.expression.col = currCol    
#     L.condition_click(None)
#     N.layout_click(None)
#     N.labels_button.value=True



