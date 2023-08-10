# download the source file if not already here
import os
fname = "constructionoutputdecember2016.xls"
if not os.path.exists(fname):
    import urllib.request
    furl = "https://www.ons.gov.uk/file?uri=/businessindustryandtrade/constructionindustry/datasets/outputintheconstructionindustry/current/constructionoutputdecember2016.xls"
    print("Downloading from", furl)
    k = urllib.request.URLopener(headers={'User-Agent' : "Sensible code"})
    k.addheader('User-Agent', "Sensible code")
    k.retrieve(furl, fname)

# Load up our construction output table from the url page:
# durl = "http://www.ons.gov.uk/businessindustryandtrade/constructionindustry/datasets/outputintheconstructionindustry"

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
from databaker.framework import *
import pandas as pd, numpy as np, re

previewhtml = "preview.html"
fname = "constructionoutputdecember2016.xls"
tabs = loadxlstabs(fname)

# Get the tables which match the most common format and the titles
import re
htabs = [ ]
for tab in tabs:
    if tab.excel_ref("C6").value == "New Housing":
        htabs.append(tab)
        print(tab.name, "\t", re.sub("\s+", " ", tab.excel_ref("C2").value))

savepreviewhtml(htabs[0], previewhtml)


# spread those multi-cell heading values across widthwise and move vertical ones down
# (should be a function within the library)

def FixMultipleCellsHeading(tab):
    hh = tab.excel_ref("C6:P9").is_not_whitespace()
    for cc in hh:
        val = cc.value
        ch = set(cc.children().unordered_cells)
        xwidth = len(set(c.x  for c in ch))
        ywidth = len(set(c.y  for c in ch))
        if ywidth == 1:
            for c in ch:
                c.value = val
        elif xwidth == 1:
            bottomcell = max(ch, key=lambda X:X.y)
            for c in ch:
                c.value = ''
            bottomcell.value = val
            

FixMultipleCellsHeading(htabs[3])
savepreviewhtml(htabs[3], previewhtml)

# Build up the conversion segment
tab = htabs[0]
period = tab.filter("Period")
savepreviewhtml(period, previewhtml)
# etc

# select the complete conversion segments
def CSegNormal(tab):
    period = tab.filter("Period")
    s4headings = period.shift((0,3)).fill(RIGHT).is_not_whitespace()
    s4up = s4headings.shift((0,-1))
    s4up2 = s4headings.shift((0,-3))
    s4up3 = s4headings.shift((0,-4))
    obs = s4headings.fill(DOWN).is_not_whitespace()
    subyear = s4headings.by_index(1).shift(LEFT).fill(DOWN)
    year = subyear.shift(LEFT).is_not_whitespace()

    hdimheadings = HDim(s4headings, "s4heading", DIRECTLY, UP)
    hdimdesc1 = HDim(s4up, "desc1", DIRECTLY, UP)
    hdimdesc2 = HDim(s4up2, "desc2", DIRECTLY, UP)
    hdimdesc3 = HDim(s4up3, "desc3", DIRECTLY, UP, LEFT)
    hdimyear = HDim(year, "year", CLOSEST, UP)
    hdimsubyear = HDim(subyear, "subyear", DIRECTLY, LEFT)

    cs = ConversionSegment(obs, [hdimheadings, hdimdesc1, hdimdesc2, hdimdesc3, hdimsubyear, hdimyear])
    return cs

#cs = CSegNormal(htabs[0])
#savepreviewhtml(cs, previewhtml)


# Convert a conversion segment to pandas
tab = htabs[3]
FixMultipleCellsHeading(tab)
cs = CSegNormal(tab)
k = cs.topandas()
#k.subyear + " " + k.year
k.tail()

# Full pandas conversion function with date fixing and heading concatenation
def ConvertPD(cs):
    k = cs.topandas()
    k["TIME"] = k.subyear + " " + k.year
    pdguessforceTIMEUNIT(k)
    k = k.drop(["subyear", "year"], axis=1)
    k.TIME = pd.to_datetime(k.apply(lambda row: row.TIME  if "Q" not in row.TIME else row.TIME.replace(" ", ""), axis=1))
    def rj(r):
        rs = [ ]
        if r.desc1:  rs.append(r.desc1.replace("-", "").strip())  # inconsistent Infra-struc-ture
        if r.desc2:  rs.append(r.desc2.strip())
        if r.desc3:  rs.append(r.desc3.strip())
        return ",".join(rs)
    k["desc"] = k.apply(lambda r: rj(r), axis=1)
    k = k.drop(["desc1", "desc2", "desc3", "s4heading"], axis=1)
    k = k.sort_values(by=["desc", "TIMEUNIT", "TIME"])
    k = k.set_index(["desc", "TIMEUNIT", "TIME"])
    return k 



# Run the full conversion across each of our tabs we've found
pds = [ ]
for tab in htabs:
    FixMultipleCellsHeading(tab)
    cs = CSegNormal(tab)
    k = ConvertPD(cs)
    pds.append(k)
    

# Now put these into meaningful variables (note the chaos)

consindexSA, consindexNSA = pds[0], pds[1]
consvolumeSA, consvolumeNSA = pds[2], pds[3]
conspercchangePeriodSA, conspercchangeYearSA = pds[4], pds[5]
consvalueNSA, consvalueSA = pds[6], pds[7]
conspriceindicNSA = pds[8]
    
#savepreviewhtml(consindexSA, previewhtml)

d = "All New Work"
k = consvolumeNSA

# look at the different levels
k.index.levels[1]

# we can get out a single element as a triple-wise index (but handling the Timestamp is a nuisance)
# (See timestamp documentation)
s = k.iloc[0]
s.name
#k.loc[(d, 'Year', pd.Timestamp('2010-01-01 00:00:00'))]

k.xs(key=d, level=0)
#d
k.index.levels[0]

k = consvolumeNSA

k.xs(key='Public,New Housing', level=0)  # <-- DataFrame
k.xs(key='Public,New Housing', level=0).OBS  # <-- Series

# show the dataframe
pd.DataFrame(data={"col1":k.xs(key='Public,New Housing', level=0).OBS, 
                   "col2":k.xs(key='Private,New Housing', level=0).OBS, 
                   "col3":k.xs(key='Total Housing', level=0).OBS}) 

# can add DataFrames (lined up by their indexes)
k.xs(key='Public,New Housing', level=0)+k.xs(key='Private,New Housing', level=0)-k.xs(key='Total Housing', level=0)

# Prove that the columns add up
k = consvalueNSA
THerr = (k.xs(key='Public,New Housing', level=0)+k.xs(key='Private,New Housing', level=0)-k.xs(key='Total Housing', level=0))
max(np.abs(THerr.OBS))

k = consvolumeNSA
k = consvalueNSA

TNWerr = k.xs('Infrastructure,Other New Work')+k.xs('Private Commercial,Excluding Infrastructure,Other New Work')         +k.xs('Private Industrial,Excluding Infrastructure,Other New Work')+k.xs('Public,Excluding Infrastructure,Other New Work')         -k.xs('All New Work')+k.xs('Total Housing')
TNWerr
max(np.abs(TNWerr.OBS))

k = consvolumeNSA
TRMerr = k.xs('Public,Housing,Repair and Maintenance')+k.xs('Private,Housing,Repair and Maintenance')         -k.xs('Total,Housing,Repair and Maintenance')
TRMerr
max(np.abs(TRMerr.OBS))

# Note bug of missing ,Housing, heading extension in Table 4
k = consvalueNSA
TRMerr = k.xs('Public,Housing,Repair and Maintenance')+k.xs('Private,Housing,Repair and Maintenance')         -k.xs('Total,Repair and Maintenance')
TRMerr
max(np.abs(TRMerr.OBS))


k = consvolumeNSA

TRMerr = k.xs('Non Housing R&M,Repair and Maintenance')+k.xs('Total,Housing,Repair and Maintenance')         -k.xs('All Repair and Maintenance')
max(np.abs(TRMerr.OBS))

k = consvalueNSA

TRMerr = k.xs('Total,Repair and Maintenance')+k.xs('Infrastructure,Repair and Maintenance')         +k.xs('Public,Other Work,Repair and Maintenance')+k.xs('Private,Other Work,Repair and Maintenance')         -k.xs('All Repair and Maintenance')
max(np.abs(TRMerr.OBS))

k = consvolumeNSA
#k = consvalueNSA
TRMerr = k.xs('All New Work')+k.xs('All Repair and Maintenance')         -k.xs('All Work')
max(np.abs(TRMerr.OBS))
#'', 'All Repair and Maintenance', 'All Work',
#'Non Housing R&M,Repair and Maintenance',
#k.index.levels[0]

k = consvolumeNSA
d = 'Public,New Housing'
s = k.xs((d, "Month")).resample("QS").apply(np.sum) - k.xs((d, "Quarter"))
s = k.xs((d, "Quarter")).resample("AS").apply(np.sum) - k.xs((d, "Year"))
"worst disagreement", max(np.abs(s.dropna().OBS))


# check resamples
k = consvolumeNSA.OBS.unstack(level=0)
s = k.xs("Month").resample("QS").apply(np.sum) - k.xs("Quarter")
s = k.xs("Quarter").resample("AS").apply(np.sum) - k.xs("Year")
s = s.dropna()
max(max(s.iloc[i])  for i in range(len(s)))

# Note that there is a disagreement here with the year summary
k = consvalueNSA.OBS.unstack(level=0)
#s = k.xs("Month").resample("QS").apply(np.sum) - k.xs("Quarter")
#s = k.xs("Quarter").resample("AS").apply(np.sum) - k.xs("Year")
s = s.dropna()

#k.xs("Quarter").to_csv("UKconstructionvaluequarter.csv")

# Prove the index values by dividing one by other
k = consvolumeNSA.OBS.unstack(level=0).xs("Month")
kI = consindexNSA.OBS.unstack(level=0).xs("Month")
print("ratios will be constant in each column")
(k/kI).head()


# what is seasonal adjustment?  How is it different in year?
k = consvalueNSA.OBS.unstack(level=0).xs("Year")
kSA = consvalueSA.OBS.unstack(level=0).xs("Year")
d = "Public,New Housing"
s = (kSA-k).dropna()
max(max(s.iloc[i])  for i in range(len(s)))

#k = consvalueNSA.OBS.unstack(level=0).xs("Quarter")
#kSA = consvalueNSA.OBS.unstack(level=0).xs("Quarter")
#kSA[d].plot()
#k[d].plot()
#(k-kSA)[d].plot()

# goes back to 1955, though you can't find it in the spreadsheet
d = "All Work"
kV = consvalueNSA.OBS.unstack(level=0).xs("Quarter")
kV[d].plot()

# what is different ratios for the years and months
k = consvolumeNSA.OBS.unstack(level=0).xs("Year")
kSA = consvolumeSA.OBS.unstack(level=0).xs("Year")
kV = consvalueNSA.OBS.unstack(level=0).xs("Year")
kVSA = consvalueSA.OBS.unstack(level=0).xs("Year")

#(k/kV)["All New Work"].dropna()
d = "All Work"
#kV[d].plot()
dn = "All New Work"
#kV[dn].plot()
#(kV[dn]/kV[d]).plot()
#k[d].plot()
#kSA[d].plot()
#kVSA[d].plot()
kV.head()

#(kV["Public,New Housing"]/kV["Total Housing"]).plot()
kV = consvalueNSA.OBS.unstack(level=0).xs("Quarter")
kVSA = consvalueSA.OBS.unstack(level=0).xs("Quarter")
#kV["Public,New Housing"].plot()
#kV["Total Housing"].plot()

#(kV["Public,New Housing"]).plot()
#(kV["Total Housing"]).plot()

#kV = consvalueNSA.OBS.unstack(level=0).xs("Month")
#kVSA = consvalueSA.OBS.unstack(level=0).xs("Month")
(kVSA[d]/kV[d]).plot()

if False:
    d = "Private,New Housing"
    #kVSA[d].plot()
    #kV[d].plot()
    
    s = kV[d] - kVSA[d]
    s.dropna().plot()
    (kV[d]*0.1).plot()




# Bovis Homes Group turnover per year copied from annual reports to see if it follows the same pattern 
# as the house building surveys
# (Could also cross-check with the DfT roadbuilding budget re infrastructure)

# http://www.bovishomesgroup.co.uk/investor-centre/reports-and-presentations/reports/view-year-2011/
bh = """2015  946
2014  809
2013  556
2012  425
2011  364
2010  298
2009  281
2008  282
2007  555
2006  597
2005  521
2004  559
2003  478
2002  461""".split()
s = [{ "TIME":y, "bovis":to } for y, to in zip(bh[::2], map(float, bh[1::2]))]
b = pd.DataFrame.from_dict(list(reversed(s)))
b.TIME = pd.to_datetime(b.TIME)
b.set_index("TIME", inplace=True)
kV = consvalueNSA.OBS.unstack(level=0).xs("Year")
#kV["Private,New Housing"].loc[pd.Timestamp("2002"):].plot()
#b.plot()

s = kV["Private,New Housing"].loc[pd.Timestamp("2001"):]/b.bovis
s.plot()
s = kV["Private,New Housing"].loc[pd.Timestamp("2001"):]/b.bovis.shift()
s.plot()



# Everything you see is in these tables (except cars and rolling stock)
# What we miss is the rate of destruction

# Move our models up to a greater level of sophistication so 
# it can be fed into predictive models and set policy.




d = "Infrastructure,Other New Work"
kV = consvalueNSA.OBS.unstack(level=0).xs("Month")
kVSA = consvalueSA.OBS.unstack(level=0).xs("Month")
(kVSA[d]/kV[d]).plot()


# save the data for some artima experiments -- understanding the seasonal adjustment technology
#import pickle
#pickle.dump((consvalueNSA, consvalueSA), open("pfile", "wb"))



