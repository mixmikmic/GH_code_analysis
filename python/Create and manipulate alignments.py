from pyrna.db import Rfam
rfam = Rfam(use_website = True)
rnas, species, consensus = rfam.get_entry(rfam_id = 'RF00058', nse_labels = 0)

for rna in rnas:
    print rna.name

consensus

