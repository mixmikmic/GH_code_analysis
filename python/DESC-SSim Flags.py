# First lets get a source catalog to play with:
# source /global/common/cori/contrib/lsst/lsstDM/setupStack-13_0_c.sh
# setup lsst_distrib

import lsst.daf.persistence as dafPersist

butler = dafPersist.Butler('/global/cscratch1/sd/descdm/DC1/full_focalplane_undithered')
srcCat = butler.get('deepCoadd_meas', filter='r', tract=0, patch='7,9')
srcCat.getSchema()

srcCat.get("deblend_nChild")
srcCat.get("detect_isPrimary")
srcCat.get("parent")

# Limit to primary like:
idx = srcCat.get("detect_isPrimary") 
print len(srcCat.get("base_PsfFlux_flux")[idx])


srcCat.get("base_PsfFlux_flag")

# NOT a DM-endorsed way of working with tables:

import pandas as pd

df = pd.DataFrame()
for col in srcCat.getSchema():
    name = col.field.getName()
    df[name] = icsrc.get(name)

df.to_csv('filename.csv')



