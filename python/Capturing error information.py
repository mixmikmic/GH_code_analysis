from rdkit import Chem
from rdkit import rdBase
print(rdBase.rdkitVersion)

m = Chem.MolFromSmiles('CO(C)C')
m

from rdkit.Chem.Draw import IPythonConsole

m = Chem.MolFromSmiles('CO(C)C')

Chem.MolFromSmiles('c1cc1')

Chem.MolFromSmiles('c1')

Chem.MolFromSmiles('Ch')

from io import StringIO
import sys
Chem.WrapLogs()

sio = sys.stderr = StringIO()
Chem.MolFromSmiles('Ch')
print("error message:",sio.getvalue())

def readmols(suppl):
    ok=[]
    failures=[]
    sio = sys.stderr = StringIO()
    for i,m in enumerate(suppl):
        if m is None:
            failures.append((i,sio.getvalue()))
            sio = sys.stderr = StringIO() # reset the error logger
        else:
            ok.append((i,m))
    return ok,failures

import gzip,os
from rdkit import RDConfig
inf = gzip.open(os.path.join(RDConfig.RDDataDir,'PubChem','Compound_000200001_000225000.sdf.gz'))
suppl = Chem.ForwardSDMolSupplier(inf)
ok,failures = readmols(suppl)

for i,fail in failures:
    print(i,fail)



