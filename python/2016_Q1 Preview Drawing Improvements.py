from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import rdBase
print(rdBase.rdkitVersion)
import time
print(time.asctime())

m = Chem.MolFromSmiles('Cc1ccccc1c2nsc(SCC(=O)Nc3ccc(Br)cc3)n2') # molecule from the 2014 TDT challenge
m

IPythonConsole.ipython_useSVG=True

m

# molecules from the 2014 TDT challenge
d=[('SJ000241686-1', 'Cc1ccccc1c2nsc(SCC(=O)Nc3ccc(F)cc3F)n2'),
 ('SJ000241766-1', 'O=C(Nc1nc(cs1)c2ccccn2)c3ccc(cc3)S(=O)(=O)N(CCC#N)CCC#N'),
 ('SJ000241694-1', 'Cc1ccccc1c2nsc(SCC(=O)Nc3ccccc3C(F)(F)F)n2'),
 ('SJ000241774-1', 'COc1cc2ccccc2cc1C(=O)\\N=C\\3/Sc4cc(Cl)ccc4N3C'),
 ('SJ000241702-1', 'CC(=O)c1ccc(NC(=O)CSc2nc(ns2)c3ccccc3C)cc1'),
 ('SJ000241785-1', 'CC1=CC=CN2C(=O)C(=C(Nc3ccc(cc3)[N+](=O)[O-])N=C12)C=O'),
 ('SJ000241710-1', 'Fc1ccccc1NC(=O)CSc2nc(ns2)c3ccccc3Cl'),
 ('SJ000241792-1', 'NC(=O)c1c2CCCCc2sc1NC(=O)\\C(=C/c3ccc(Cl)cc3)\\C#N'),
 ('SJ000241718-1', 'COc1cccc(NC(=O)CSc2nc(ns2)c3ccccc3Cl)c1'),
 ('SJ000241800-1',
  'NC(=O)c1c2CCCCc2sc1NC(=O)\\C(=C\\c3ccc(cc3)[N+](=O)[O-])\\C#N'),
 ('SJ000241726-1', 'FC(F)(F)c1ccc(NC(=O)CSc2nc(ns2)c3ccccc3Cl)cc1'),
 ('SJ000241808-1', 'COc1ccc(\\C=C(\\C#N)/C(=O)Nc2cccc(c2)C(F)(F)F)cc1OC')]
ms = [Chem.MolFromSmiles(y) for x,y in d]
labels = [x for x,y in d]
Draw.MolsToGridImage(ms,legends=labels,molsPerRow=4)

ms[1].GetSubstructMatches(Chem.MolFromSmarts('CC#N'))
ms[1]

from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor

m = Chem.MolFromSmiles('COc1cccc(NC(=O)[C@H](Cl)Sc2nc(ns2)c3ccccc3Cl)c1') # something I made up
rdDepictor.Compute2DCoords(m)
d2d = rdMolDraw2D.MolDraw2DSVG(300,250)
d2d.DrawMolecule(m)
d2d.FinishDrawing()
svg = d2d.GetDrawingText()
SVG(svg.replace("svg:",""))

m = Chem.MolFromSmiles('COc1cccc(NC(=O)[C@H](Cl)Sc2nc(ns2)c3ccccc3Cl)c1') # something I made up
tm = rdMolDraw2D.PrepareMolForDrawing(m)
d2d = rdMolDraw2D.MolDraw2DSVG(300,250)
d2d.DrawMolecule(tm)
d2d.FinishDrawing()
svg = d2d.GetDrawingText()
SVG(svg.replace("svg:",""))

m = Chem.MolFromSmiles('COc1cccc(NC(=O)[C@H](Cl)Sc2nc(ns2)c3ccccc3Cl)c1') # something I made up
m

estradiol=Chem.MolFromSmiles('C[C@]12CC[C@@H]3c4ccc(cc4CC[C@H]3[C@@H]1CC[C@@H]2O)O')
estradiol

om = Chem.MolFromSmiles('C[C@]12CC[C@@H]3c4ccc(Br)c(=O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O')
om



