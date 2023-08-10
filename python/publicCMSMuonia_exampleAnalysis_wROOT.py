sc.defaultParallelism

df = sqlContext.read.format("org.dianahep.sparkroot").option("tree", "Events").load("hdfs:/cms/bigdatasci/vkhriste/data/publiccms_muionia_aod")
#df1 = sqlContext.read.format("org.dianahep.sparkroot").option("tree", "Events").load("hdfs:/cms/bigdatasci/vkhriste/data/publiccms_muionia_aod/0000/FEEFB039-0978-E011-BB60-E41F131815BC.root")
df.printSchema()

df.count()

slimmedEvents = df.select("recoMuons_muons__RECO_.recoMuons_muons__RECO_obj.reco::RecoCandidate.reco::LeafCandidate")
slimmedEvents.printSchema()

slimmedEvents.show()

from math import *
import numpy
from pyspark.sql import Row

def invariantMass(muon1, muon2):
    pt1 = abs(muon1.pt_)
    phi1 = muon1.phi_
    theta1 = 2.0*atan(exp(-muon1.eta_))
    px1 = pt1 * cos(phi1)
    py1 = pt1 * sin(phi1)
    pz1 = pt1 / tan(theta1)
    E1 = sqrt(px1**2 + py1**2 + pz1**2 + 0.10565836727619171**2)
    #
    pt2 = abs(muon2.pt_)
    phi2 = muon2.phi_
    theta2 = 2.0*atan(exp(-muon2.eta_))
    px2 = pt2 * cos(phi2)
    py2 = pt2 * sin(phi2)
    pz2 = pt2 / tan(theta2)
    E2 = sqrt(px2**2 + py2**2 + pz2**2 + 0.10565836727619171**2)
    #
    return sqrt((E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2)
def handleEvent(event):
    # decreasing order of muon pT
    sortedMuons = sorted(event[0], key=lambda muon: -muon.pt_)
    if len(sortedMuons) < 2:
        return []
    else:
        muon1, muon2 = sortedMuons[:2]
        # return [Row(mass=invariantMass(muon1, muon2), pt1=muon1.pt_, phi1=muon1.phi_, eta1=muon1.eta_, pt2=muon2.pt_, phi2=muon2.phi_, eta2=muon2.eta_)]
        return [invariantMass(muon1, muon2)]

# 
# doing toDF.persist directly is not working
# http://stackoverflow.com/questions/32742004/create-spark-dataframe-can-not-infer-schema-for-type-type-float/32742294
#
dimuon_masses = slimmedEvents.rdd.flatMap(handleEvent).map(lambda x: (x, )).toDF().persist()

from histogrammar import *

#from primitives.bin import Bin

empty = Bin(100, 0, 10, lambda x: x._1)
filled = dimuon_masses.rdd.aggregate(empty, increment, combine)

get_ipython().magic('matplotlib inline')
filled.plot.matplotlib(name="Dimuon Mass")

import ROOT
hist = filled.plot.root("Dimuon Mass", "DimuonMass")
c = ROOT.TCanvas("c")
hist.Draw()
c.Draw()

df2 = sqlContext.read.format("org.dianahep.sparkroot").load("hdfs:/cms/bigdatasci/vkhriste/data/higgs")
df2.printSchema()

df2.show()

