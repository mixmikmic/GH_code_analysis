from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG=True
from rdkit import rdBase
print(rdBase.rdkitVersion)
import time,gzip
print(time.asctime())

FilterCatalogParams.FilterCatalogs.names

params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)

filters = FilterCatalog.FilterCatalog(params)

# start with a convenience function that finds a molecule matching the
# FilterCatalog passed in and returns the matching molecule as well
# as the match information
def findMatch(fc):
    inf = gzip.open("../data/malariahts_trainingset.txt.gz") # HTS data from the 2014 TDT challenge
    keep = []
    nReject=0
    inf.readline() # ignore the header line
    for i,line in enumerate(inf):
        splitL = line.strip().split()
        smi = splitL[-1]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        if fc.HasMatch(m):
            matches = fc.GetMatches(m)
            break
    return m,matches

m,matches = findMatch(filters)
for match in matches:
    print(match.GetDescription())
    print(match.GetProp("Reference"))
    print(match.GetProp("Scope"))
    print("------------")
m

hatoms = [x[1] for x in matches[1].GetFilterMatches(m)[0].atomPairs]
# take advantage of the way the Jupyter notebook integration works:
m.__sssAtoms=hatoms
m

hatoms = [x[1] for x in matches[0].GetFilterMatches(m)[0].atomPairs]
m.__sssAtoms=hatoms
m

m.__sssAtoms=None

two_phenyls = FilterCatalog.SmartsMatcher("at least two phenyls","[cD3]1[cD2][cD2][cD2][cD2][cD2]1",2)
fc = FilterCatalog.FilterCatalog()
fc.AddEntry(FilterCatalog.FilterCatalogEntry("matcher",two_phenyls))

m,matches = findMatch(fc)
m

thiazole = FilterCatalog.SmartsMatcher("thiazole","c1sccn1",1)
combined = FilterCatalog.FilterMatchOps.And(two_phenyls,FilterCatalog.FilterMatchOps.Not(thiazole))
fc = FilterCatalog.FilterCatalog()
fc.AddEntry(FilterCatalog.FilterCatalogEntry("matcher",combined))


m,matches = findMatch(fc)
m

excluder = FilterCatalog.ExclusionList()
excluder.AddPattern(FilterCatalog.SmartsMatcher("thiazole","c1sccn1",1))
excluder.AddPattern(FilterCatalog.SmartsMatcher("thiophene","c1sccc1",1))
excluder.AddPattern(FilterCatalog.SmartsMatcher("piperazine","C1CNCCN1",1))

combined = FilterCatalog.FilterMatchOps.And(two_phenyls,excluder)
fc = FilterCatalog.FilterCatalog()
fc.AddEntry(FilterCatalog.FilterCatalogEntry("matcher",combined))

m,matches = findMatch(fc)
m

from rdkit.Chem import Descriptors
class MWFilter(FilterCatalog.FilterMatcher):
    def __init__(self, minMw, maxMw):
        FilterCatalog.FilterMatcher.__init__(self, "MW violation")
        self.minMw = minMw
        self.maxMw = maxMw

    def IsValid(self):
        return True

    def HasMatch(self, mol):
        mw=Descriptors.MolWt(mol)
        return self.minMw <= mw <= self.maxMw

# find smallish molecules:
mw = MWFilter(300,400)
combined = FilterCatalog.FilterMatchOps.And(two_phenyls,mw)
fc = FilterCatalog.FilterCatalog()
fc.AddEntry(FilterCatalog.FilterCatalogEntry("matcher",combined))

m,matches = findMatch(fc)
print(Descriptors.MolWt(m))
m

