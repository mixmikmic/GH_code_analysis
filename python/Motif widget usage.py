# allow import without install in Binder
import sys
if ".." not in sys.path:
    sys.path.append("..")

from IPython.display import display
from jp_gene_viz import motif_data

C = motif_data.MotifCollection()
# for extra safety "rU" reads with universal line ending support
C.read_meme_file(open("mm9_em.meme", "rU"))

C.letter_order

motif_names = list(C.name_to_motif.keys())
motif_names[:20]

# Choose a particular motif by name.
Ebf1 = C["Ebf1"]

# Display the motif logo with entropy.
Ebf1_with_entropy = Ebf1.canvas()
display(Ebf1_with_entropy)

# Display the motif logo without entropy.
Ebf1_no_entropy = Ebf1.canvas(entropy=False)
display(Ebf1_no_entropy)

# display 10 randomly chosen motifs with and without entropy
import random
for i in range(10):
    name = random.choice(motif_names)
    mt = C[name]
    c = mt.canvas(entropy=False)
    display(c)
    c.evaluate(c.element().after("<div>%s</div" % name))
    c = mt.canvas()
    display(c)
    c.evaluate(c.element().after("<div>%s with entropy</div" % name))

# First create a (fake) network data file which includes the network motif
# comma separated list as a fourth column.  For the fake data we assign motifs
# randomly.
from jp_gene_viz import getData
n = getData.read_network("network.tsv")
out_file_name = "network_with_motifs.tsv"
out_file = open(out_file_name, "w")
out_file.write("Regulator\tTarget\tWeight\tMotifs\n")
ew = n.edge_weights
for e in ew:
    (r, t) = e
    w = str(ew[e])
    nmotifs = random.randint(0,5)
    # add the ignored suffix here
    motifs = set(random.choice(motif_names) + "_hg19" for i in range(nmotifs))
    m = ",".join(list(motifs))
    out_file.write(("\t".join([r, t, w, m])) + "\n")
out_file.close()

# Display the first few rows of the faked out data file.
print(open(out_file_name).read(500))

# Load the implementation
from jp_gene_viz import dNetwork
#reload(dNetwork)
dNetwork.load_javascript_support()

# Read the network data file with the motif column.
net_with_motifs = dNetwork.display_network(out_file_name)

# Attach the motif collection populated above:
net_with_motifs.motif_collection = C

# display the network with motifs.
net_with_motifs

