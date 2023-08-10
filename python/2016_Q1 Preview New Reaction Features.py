from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG=True
from rdkit import rdBase
print(rdBase.rdkitVersion)
import time
print(time.asctime())

rxn = AllChem.ReactionFromSmarts('[cH1:1]1:[c:2](-[CH2:7]-[CH2:8]-[NH2:9]):[c:3]:[c:4]:[c:5]:[c:6]:1.'
                                 '[#6:11]-[CH1;R0:10]=[OD1]>>'
                                 '[c:1]12:[c:2](-[CH2:7]-[CH2:8]-[NH1:9]-[C:10]-2(-[#6:11])):[c:3]:[c:4]:[c:5]:[c:6]:1	')
rxn

reactants = [Chem.MolFromSmiles(x) for x in ('c1cc(OC(F)(F)F)ccc1CCN','C1CC1CC=O')]
Draw.MolsToGridImage(reactants)

ps = rxn.RunReactants(reactants)
ps[0][0]

p0s = rxn.RunReactant(reactants[0],0)
p0s[0][0]

p1s = rxn.RunReactant(reactants[1],1)
p1s[0][0]

AllChem.ReduceProductToSideChains(p0s[0][0],addDummyAtoms=True)

AllChem.ReduceProductToSideChains(p0s[0][0],addDummyAtoms=False)

AllChem.ReduceProductToSideChains(p1s[0][0],addDummyAtoms=True)

AllChem.ReduceProductToSideChains(p1s[0][0],addDummyAtoms=False)

AllChem.ReduceProductToSideChains(ps[0][0],addDummyAtoms=True)

f_rxn = AllChem.ReactionFromSmarts('[NH2;$(N-c1ccccc1):1]-[c:2]:[c:3]-[CH1:4]=[OD1].'
                                    '[C;$(C([#6])[#6]):6](=[OD1])-[CH2;$(C([#6])[#6]);!$(C(C=O)C=O):5]>>'
                                    '[N:1]1-[c:2]:[c:3]-[C:4]=[C:5]-[C:6]:1')
f_rxn

f_reactants = [Chem.MolFromSmiles(x) for x in ['Nc1c(C=O)ccc(OC)c1','ClCCC(=O)CBr']]
Draw.MolsToGridImage(f_reactants)

ps = f_rxn.RunReactants(f_reactants)
ps[0][0]

p0s = f_rxn.RunReactant(f_reactants[0],0)
p0s[0][0]

p1s = f_rxn.RunReactant(f_reactants[1],1)
p1s[0][0]

AllChem.ReduceProductToSideChains(p0s[0][0],addDummyAtoms=True)

AllChem.ReduceProductToSideChains(p1s[0][0],addDummyAtoms=True)

AllChem.ReduceProductToSideChains(ps[0][0],addDummyAtoms=True)

f_rxn2 = AllChem.ReactionFromSmarts('[NH2:1]-[c:2]1:[c:3](-[CH1:4]=[OD1])[c:7][c:8][c:9][c:10]1.'
                                    '[C;$(C([#6])[#6]):6](=[OD1])-[CH2;$(C([#6])[#6]);!$(C(C=O)C=O):5]>>'
                                    '[N:1]1-[c:2]2:[c:3](-[C:4]=[C:5]-[C:6]:1)[c:7][c:8][c:9][c:10]2')
f_rxn2

ps = f_rxn2.RunReactants(f_reactants)
ps[0][0]

p0s = f_rxn2.RunReactant(f_reactants[0],0)
p0s[0][0]

p1s = f_rxn2.RunReactant(f_reactants[1],1)
p1s[0][0]

AllChem.ReduceProductToSideChains(p0s[0][0],addDummyAtoms=True)

AllChem.ReduceProductToSideChains(p1s[0][0],addDummyAtoms=True)

