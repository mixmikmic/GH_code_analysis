import glob
import pandas as pd
for c in ["ttbar_lepFilter_13TeV", "wjets_lepFilter_13TeV", "qcd_lepFilter_13TeV"]:
    files = glob.glob("/data/shared/Delphes/%s/pandas_h5/*.h5" % c)
    f = files[0]
    store = pd.HDFStore(f)
    for obj in ["EFlowPhoton", "EFlowNeutralHadron", "EFlowTrack"]:
        m = []
        dataframe = store.get(obj)
        for i in range(1,500):
            df = dataframe[dataframe["Entry"] == i]
            #l1 = len(df.index)
            #print(len(df.index))
            df = df.query("PT_ET > 1.5")
            l2 = len(df.index)
            m.append(l2)
        print(c,obj,"MAX: "+str(max(m)),"AVG: " + str(sum(m)/len(m)), "MIN: "+str(min(m)))
    store.close()

store = pd.HDFStore("/data/shared/Delphes/ttbar_lepFilter_13TeV/pandas_h5/ttbar_lepFilter_13TeV_0.h5")
dataframe = store.get("NumValues")
print(dataframe)

#PT >1
('ttbar_lepFilter_13TeV', 'EFlowPhoton', 'MAX: 165', 'AVG: 82')
('ttbar_lepFilter_13TeV', 'EFlowNeutralHadron', 'MAX: 287', 'AVG: 132')
('ttbar_lepFilter_13TeV', 'EFlowTrack', 'MAX: 320', 'AVG: 158')
('wjets_lepFilter_13TeV', 'EFlowPhoton', 'MAX: 157', 'AVG: 57')
('wjets_lepFilter_13TeV', 'EFlowNeutralHadron', 'MAX: 277', 'AVG: 112')
('wjets_lepFilter_13TeV', 'EFlowTrack', 'MAX: 276', 'AVG: 116')
('qcd_lepFilter_13TeV', 'EFlowPhoton', 'MAX: 178', 'AVG: 80')
('qcd_lepFilter_13TeV', 'EFlowNeutralHadron', 'MAX: 339', 'AVG: 133')
('qcd_lepFilter_13TeV', 'EFlowTrack', 'MAX: 392', 'AVG: 155')

#PT >1.5
('ttbar_lepFilter_13TeV', 'EFlowPhoton', 'MAX: 92', 'AVG: 39')
('ttbar_lepFilter_13TeV', 'EFlowNeutralHadron', 'MAX: 167', 'AVG: 73')
('ttbar_lepFilter_13TeV', 'EFlowTrack', 'MAX: 164', 'AVG: 82')
('wjets_lepFilter_13TeV', 'EFlowPhoton', 'MAX: 65', 'AVG: 23')
('wjets_lepFilter_13TeV', 'EFlowNeutralHadron', 'MAX: 163', 'AVG: 58')
('wjets_lepFilter_13TeV', 'EFlowTrack', 'MAX: 125', 'AVG: 49')
('qcd_lepFilter_13TeV', 'EFlowPhoton', 'MAX: 78', 'AVG: 38')
('qcd_lepFilter_13TeV', 'EFlowNeutralHadron', 'MAX: 200', 'AVG: 73')
('qcd_lepFilter_13TeV', 'EFlowTrack', 'MAX: 170', 'AVG: 78')

#No Cut
('ttbar_lepFilter_13TeV', 'EFlowPhoton', 'MAX: 1016', 'AVG: 532', 'MIN: 208')
('ttbar_lepFilter_13TeV', 'EFlowNeutralHadron', 'MAX: 816', 'AVG: 413', 'MIN: 156')
('ttbar_lepFilter_13TeV', 'EFlowTrack', 'MAX: 828', 'AVG: 419', 'MIN: 158')
('wjets_lepFilter_13TeV', 'EFlowPhoton', 'MAX: 1051', 'AVG: 472', 'MIN: 122')
('wjets_lepFilter_13TeV', 'EFlowNeutralHadron', 'MAX: 759', 'AVG: 376', 'MIN: 84')
('wjets_lepFilter_13TeV', 'EFlowTrack', 'MAX: 809', 'AVG: 357', 'MIN: 86')
('qcd_lepFilter_13TeV', 'EFlowPhoton', 'MAX: 1168', 'AVG: 542', 'MIN: 192')
('qcd_lepFilter_13TeV', 'EFlowNeutralHadron', 'MAX: 937', 'AVG: 421', 'MIN: 135')
('qcd_lepFilter_13TeV', 'EFlowTrack', 'MAX: 985', 'AVG: 423', 'MIN: 140')

