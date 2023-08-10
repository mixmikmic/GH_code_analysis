from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import rdBase
get_ipython().magic('load_ext sql')
print(rdBase.rdkitVersion)
import time
print(time.asctime())

data = get_ipython().magic("sql postgresql://localhost/chembl_21     select molregno,molfile from rdk.mols join compound_structures     using (molregno) where m@>'c1ccccn1' limit 10;")
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)

data = get_ipython().magic("sql     select molregno,molfile from rdk.mols join compound_structures     using (molregno) where m@>'*c1cccc(*)n1' limit 10 ;")

data = get_ipython().magic("sql     select molregno,molfile from rdk.mols join compound_structures     using (molregno) where m@>mol_adjust_query_properties('*c1cccc(*)n1') limit 10 ;")
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)    

data = get_ipython().magic('sql     select molregno,molfile from rdk.mols join compound_structures     using (molregno) where m@>mol_adjust_query_properties(\'*c1cccc(*)n1\',                                                         \'{"adjustDegree"\\:false}\') limit 10 ;')
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)    

data = get_ipython().magic('sql     select molregno,molfile from rdk.mols join compound_structures     using (molregno) where m@>mol_adjust_query_properties(\'*c1cccc(*)n1\',                                                         \'{"adjustDegree"\\:false,                                                          "adjustRingCount"\\:true}\') limit 10 ;')
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)    

data = get_ipython().magic("sql     select molregno,molfile from rdk.mols join compound_structures     using (molregno) where m@>mol_adjust_query_properties('*c1cccc(NC(=O)*)n1') limit 10 ;")
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)    

data = get_ipython().magic('sql     select molregno,molfile from rdk.mols join compound_structures     using (molregno) where m@>mol_adjust_query_properties(\'*c1cccc(NC(=O)*)n1\',                                                         \'{"adjustDegree"\\:true,                                                          "adjustDegreeFlags"\\:"IGNORERINGS|IGNOREDUMMIES"                                                          }\') limit 10 ;')
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)    

data = get_ipython().magic('sql     select molregno,molfile from rdk.mols join compound_structures     using (molregno) where m@>mol_adjust_query_properties(\'*c1cccc(NC(=O)*)n1\',                                                         \'{"adjustDegree"\\:true,                                                          "adjustDegreeFlags"\\:"IGNORERINGS|IGNOREDUMMIES",                                                          "adjustRingCount"\\:true}\') limit 10 ;')
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)    

mb="""
  Mrv1561 07261609522D          

  8  8  0  0  0  0            999 V2000
   -1.9866    0.7581    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7011    0.3455    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7011   -0.4795    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9866   -0.8920    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2721   -0.4795    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2721    0.3455    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.4155    0.7580    0.0000 A   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5577    0.7580    0.0000 A   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  6  2  0  0  0  0
  2  3  2  0  0  0  0
  3  4  1  0  0  0  0
  4  5  2  0  0  0  0
  5  6  1  0  0  0  0
  2  7  1  0  0  0  0
  6  8  1  0  0  0  0
M  END
"""
Chem.MolFromMolBlock(mb)

data = get_ipython().magic('sql     select molregno,molfile from rdk.mols join compound_structures     using (molregno) where m@>mol_from_ctab(:mb) limit 10 ;')
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)    

data = get_ipython().magic('sql     select molregno,molfile from rdk.mols join compound_structures     using (molregno) where m@>mol_adjust_query_properties(mol_from_ctab(:mb)) limit 10 ;')
mols = [Chem.MolFromMolBlock(y) for x,y in data]
Draw.MolsToGridImage(mols,legends=[str(x) for x,y in data],molsPerRow=4)    

m1 = Chem.MolFromSmiles('Cc1cc(C)nc(NC(=O)c2ccc3c(c2)OCO3)c1')
m2 = Chem.MolFromSmiles('c1cc(C)nc(NC(=O)c2ccc3c(c2)OCO3)c1')
m3 = Chem.MolFromSmiles('c1cc(C)nc(N(C)C(=O)c2ccc3c(c2)OCO3)c1')
Draw.MolsToGridImage((m1,m2,m3),legends=['m1','m2','m3'])

q = Chem.MolFromSmiles('*c1cccc(NC(=O)*)n1')
q

m1.HasSubstructMatch(q),m2.HasSubstructMatch(q),m3.HasSubstructMatch(q)

tq = Chem.AdjustQueryProperties(q)
m1.HasSubstructMatch(tq),m2.HasSubstructMatch(tq),m3.HasSubstructMatch(tq)

params = Chem.AdjustQueryParameters()
params.adjustDegree=False
tq = Chem.AdjustQueryProperties(q,params)
m1.HasSubstructMatch(tq),m2.HasSubstructMatch(tq),m3.HasSubstructMatch(tq)

params = Chem.AdjustQueryParameters()
params.adjustDegree=True
params.adjustDegreeFlags=Chem.ADJUST_IGNORERINGS|Chem.ADJUST_IGNOREDUMMIES
tq = Chem.AdjustQueryProperties(q,params)
m1.HasSubstructMatch(tq),m2.HasSubstructMatch(tq),m3.HasSubstructMatch(tq)

