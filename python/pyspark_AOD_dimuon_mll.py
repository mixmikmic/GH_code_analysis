from math import *
#import numpy as np
#from pyspark.sql import Row

filename = "file:/home/olivito/datasci/spark/data/EC8239EF-1181-E211-8953-001EC9D80AB9.root"

ds = sqlContext.read.format("org.dianahep.sparkroot.experimental").option("tree","Events").load(filename)

ds.count()

ds.printSchema()

dsMuons = ds.select("recoMuons_muons__RECO_.recoMuons_muons__RECO_obj").toDF("muons")

dsMuons.printSchema()

dsMuons.show()

dsMuons.select(dsMuons.muons.getField("pt_")).show()

# helper function to calculate isolation from the struct
def iso(isoStruct): 
    neutral = max(0.0, isoStruct.sumNeutralHadronEt + isoStruct.sumPhotonEt - 0.5 * isoStruct.sumPUPt)
    return isoStruct.sumChargedHadronPt + neutral

# just do the looser muon selection here.  will do a second selection for leading muon later
def passMuonSel(muon):
    return ((muon.pt_ > 10.0) and 
        (fabs(muon.eta_) < 2.4) and
        (iso(muon.pfIsolationR04_)/muon.pt_ < 0.5) and
        ((muon.type_ & (1<<1)) != 0)) # global muon 

# method to apply the event level selection cuts,
#  including tighter cuts on the leading muon
def passEventSel(muons):
    return ((len(muons) > 1) and
            (muons[0].pt_ > 25.0) and
            (fabs(muons[0].eta_) < 2.1) and
            (iso(muons[0].pfIsolationR04_)/muons[0].pt_ < 0.12) and
            (muons[0].pdgId_ * muons[1].pdgId_ < 0))

# simplified formula, assuming E >> m
def invariantMass(mu1, mu2):
    return sqrt(2*mu1.pt_*mu2.pt_*(cosh(mu1.eta_-mu2.eta_)-cos(mu1.phi_-mu2.phi_)))
    
def handleEvent(event):
    # first select muons
    selMuons = [muon for muon in event.muons if passMuonSel(muon)]
    # sort in decreasing order of muon pT - makes a noticeable difference in how many events pass
    # if not sorting, can reproduce the scala results exactly
    #sortedMuons = sorted(selMuons, key=lambda muon: -muon.pt_)
    sortedMuons = selMuons
    # check if event passes selection (including requiring at least 2 muons)
    if passEventSel(sortedMuons):
        return [invariantMass(sortedMuons[0], sortedMuons[1])]
        ### from victor's example:
        # muon1, muon2 = sortedMuons[:2]
        # return [Row(mass=invariantMass(muon1, muon2), pt1=muon1.pt_, phi1=muon1.phi_, eta1=muon1.eta_, pt2=muon2.pt_, phi2=muon2.phi_, eta2=muon2.eta_)]
    else:
        return []

#dsDimuons = dsMuons.rdd.filter(lambda event: len(event.muons) > 1)
#print dsDimuons.count()

# do all the steps at once.. didn't immediately see how to do object selection in a separate step
dsMll = dsMuons.rdd.flatMap(handleEvent).map(lambda x: (x, )).toDF().persist()

dsMll.show()

dsMll.count()

dsMll.write.parquet("file:/home/olivito/datasci/spark/mll_AOD_python.parquet", mode="overwrite")



