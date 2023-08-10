import sys

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('load_ext sql')
get_ipython().magic('pylab inline')
get_ipython().magic('matplotlib inline')

import pandas as pd
import pickle,time
import pymongo 

import pylab as pl

FT1=['p53Act','DNADamage','StressKinase','OxidativeStress','MicrotubuleCSK','LysosomalMass',
     'MitoMass','MitoMembPot','MitoFxnI','Steatosis',
     'MitoticArrest','CellCycleArrest','NuclearSize','DNATexture','Apoptosis','CellLoss']

# This package contains the object (i.e. schema) definition for the database

from bio.data.htsdb import *

# This pacakge contains the functions for retrieving and displaying the data 
import bio.comp.hts as hts

# Count the number of Plates 
HtsPlate.objects.count()

# Find Troglitazone
print hts.getChem('troglitazone')
# Now get the chemical object with the eid 
TGTZ = HtsChem.objects(eid='TX006205').first()

# Get the plates containing troglitazone
Plates0 = HtsPlate.objects(chems=TGTZ)
Plates0

# Which assays are on the first plate ? 
[A.name for A in Plates0[0].assays]

# Find all assays for Troglitazone
[(A.eid,A.name) for A in HtsPlate.objects(chems=TGTZ).distinct('assays')]

# Get the p53Activation assay object
P53 = HtsAssay.objects(eid='p53Act').first()

# Find the normalized concentration response data for TGTZ for the p53 endpoint
for CRCi in HtsConcRespCurveNrm.objects(chem=TGTZ,assay=P53):
    print CRCi.timeh

# Get the concentration response data as a pandas dataframe
X=hts.getChemConcResp(u'TX006205',ret='df')

# Get the percentage change data 
X[('ppct')]

# Get the assay results 
AR=hts.getChemAssayResults('TX006205')

AR[('hit_call')]

# Plot the conc response for Octanoic acid
hts.plotHtsConcRespHM('TX002151',exp_id='APR-HepG2-PhII',add_t0=True,loc=None)

FTLB3 = dict(zip(FT1,['p53','SK','OS','Mt','MM','MMP','MA','CCA','NS','CN']))
hts.plotHtsTrajHM('TX002151',exp_id='APR-HepG2-PhII',add_t0=True,cb=True,use_resp='slfc',
                  draw_chem=False,FTLB=FTLB3,
                  fs=1.1,xyoff=[2,-4],fgsz=[15,3],loc=None)

