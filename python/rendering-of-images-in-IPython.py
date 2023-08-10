import rdkit.Chem as Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import IPythonConsole # Enables RDKit IPython integration

class primitiveMolsObject():
    def __init__(self, mols=None):
        self.mols = mols
        self.num = len(mols) # Return number of mols

mol1 = Chem.MolFromSmiles('NC(=O)CS(=O)C(c1ccccc1)c1ccccc1')

mol2 = Chem.MolFromSmiles('CCC(OC(C)=O)C(CC(C)N(C)C)(c1ccccc1)c1ccccc1')

mol3 = Chem.MolFromSmiles(' Cc1ccccc1C(OCCN(C)C)c1ccccc1')

mols = [mol1, mol2, mol3]

MyMols = primitiveMolsObject(mols)

MyMols

MyMols.mols

MyMols.num

class primitiveMolsObject2():
    def __init__(self, mols=None):
        self.mols = mols
        self.num = len(mols) # Return number of mols
    
    def _repr_html_(self):
        # Default representation in IPython
        smilesString = ''
        for mol in mols:
            smilesString += Chem.MolToSmiles(mol) + ", " 
        return smilesString #'<img src="data:image/png;base64,%s" alt="Mol"/>' %s

MyMols2 = primitiveMolsObject2(mols)

MyMols2

from base64 import b64encode
from StringIO import StringIO

class primitiveMolsObject3():
    def __init__(self, mols=None):
        self.mols = mols
        self.num = len(mols) # Return number of mols
    
    def _repr_html_(self):
        # Default representation in IPython
        sio = StringIO()
        Draw.MolsToGridImage(self.mols).save(sio,format='PNG')
        s = b64encode(sio.getvalue()) # Encode in base64
        return '<img src="data:image/png;base64,%s" alt="Mol"/>' %s

MyOtherMols = primitiveMolsObject3(mols)

MyOtherMols

