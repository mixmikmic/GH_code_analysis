get_ipython().magic('pylab inline')
import pandas as pd

from IPython.display import display
from IPython.display import display_pretty, display_html, HTML, Javascript, Image

from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFMCS as MCS
from rdkit.Chem import Draw
Draw.DrawingOptions.elemDict[0]=(0.,0.,0.)  # draw dummy atoms in black

from itertools import cycle
from sklearn import manifold
from scipy.spatial.distance import *

import mpld3
mpld3.enable_notebook()

import warnings
warnings.filterwarnings('ignore')

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

# Get the base URL of the VM:
display(Javascript('IPython.notebook.kernel.execute("current_url = " + "\'"+document.URL+"\'");'))

from rdkit import rdBase
rdBase.rdkitVersion

cur_base_url = current_url.split('http://')[1].split('/')[0]
base_url = 'localhost:8000' if (cur_base_url == 'localhost:9612' or cur_base_url == '127.0.0.1:9612') else current_url.split('http://')[1].split(':')[0] + ':8000'
base_url

pd.options.display.mpl_style = 'default'
pd.options.display.float_format = '{:.2f}'.format

rcParams['figure.figsize'] = 16,10

df = pd.read_csv('/home/chembl/ipynb_workbench/document_chemistry_20141011_114421_271.csv',sep=',')

df.info()

df.shape

df['SCHEMBL ID'] = df['Chemical ID'].map(lambda x: 'SCHEMBL{0}'.format(x))
df = df.sort('Chemical ID')

df.iloc[600:610]

dff = df[(df['Claims Count'] > 0) | (df['Description Count'] > 0) | (df['Type'] != "TEXT")]

dff.shape

dff = dff[(dff['Rotatable Bound Count'] < 15) & (dff['Ring Count'] > 0) & (df['Radical'] == 0) & (df['Singleton'] == 0) & (df['Simple'] == 0)]

dff.shape

dff = dff[(dff['Molecular Weight'] >= 300) & (dff['Molecular Weight'] <= 800) & (dff['Log P'] > 0) & (dff['Log P'] < 7)]

dff.shape

dff = dff[(dff['Chemical Corpus Count'] < 400) & ((dff['Annotation Corpus Count'] < 400) | (dff['Annotation Corpus Count'].isnull()))]

dff.shape

dff['SMILES'] = dff['SMILES'].map(lambda x: max(x.split('.'), key=len))

PandasTools.AddMoleculeColumnToFrame(dff, smilesCol = 'SMILES')

#dff.head(10)

dff['InChI'] = dff['ROMol'].map(Chem.MolToInchi)
dff['InChIKey'] = dff['InChI'].map(Chem.InchiToInchiKey)

#dff.head()

dff = dff.drop_duplicates('Chemical ID')

dff.shape

dff = dff.drop_duplicates('InChIKey')

dff.shape

dff = dff.ix[~(dff['ROMol'] >= Chem.MolFromSmarts('[#5]'))]

dff.shape

dff = dff.set_index('Chemical ID', drop=False)

dff_counts = dff[['Patent ID','ROMol']].groupby('Patent ID').count()

dff_counts['Link'] = dff_counts.index.map(lambda x: '<a href="https://www.surechembl.org/document/{0}/" target="_blank">{0}</a>'.format(x))

dff_counts = dff_counts.rename(columns={'ROMol':'# Compounds'})

dff_counts

dff.ix[dff.InChIKey == 'SECKRCOLJRRGGV-UHFFFAOYSA-N',['SCHEMBL ID','ROMol']]

fps = [Chem.GetMorganFingerprintAsBitVect(m,2,nBits=2048) for m in dff['ROMol']]

dist_mat = squareform(pdist(fps,'jaccard'))

mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=3, n_jobs = 2)

results = mds.fit(dist_mat)

coords = results.embedding_

dff['X'] = [c[0] for c in coords]

dff['Y'] = [c[1] for c in coords]

csss = """
table
{
  border-collapse: collapse;
}
th
{
  color: #ffffff;
  background-color: #848482;
}
td
{
  background-color: #f2f3f4;
}
table, th, td
{
  font-family:Arial, Helvetica, sans-serif;
  border: 1px solid black;
  text-align: right;
}
"""

import base64

dff['base64smi'] = dff['SMILES'].map(base64.b64encode)

fig, ax = plt.subplots()
fig.set_size_inches(14.0, 12.0)
#colors = cycle('bgrcmykwbgrcmykbgrcmykwbgrcmykw')
colors = cycle(cm.Dark2(np.linspace(0,1,len(dff_counts.index))))
for name, group in dff.groupby('Patent ID'):
    labels = []
    for i in group.index:
        zz = group.ix[[i],['Patent ID','SCHEMBL ID','base64smi']]
        zz['mol'] = zz['base64smi'].map(lambda x: '<img src="http://{0}/utils/smiles2image/{1}?size=300">'.format(base_url,x))
        del zz['base64smi']
        label = zz.T
        del zz
        label.columns = ['Row: {}'.format(i)]
        labels.append(str(label.to_html()))
    points = ax.scatter(group['X'], group['Y'],c=colors.next(), s=80, alpha=0.8)
    tooltip = mpld3.plugins.PointHTMLTooltip(points, labels, voffset=10, hoffset=10, css=csss)
    mpld3.plugins.connect(fig,tooltip)                                          

resmarts = "([R:5][C;!R:1]=[C;H2].[C;H0,H1;!R:4]-[N;H2:2])>>[R:5][*:1]=[*:2]-[*:4]"

Draw.ReactionToImage(Chem.ReactionFromSmarts(resmarts))

def fixRing(mol,resmarts=resmarts):
    rxn = Chem.ReactionFromSmarts(resmarts)
    ps = rxn.RunReactants((mol,))
    if ps:
        m = ps[0][0]
        Chem.SanitizeMol(m)
        return m
    else:
        return mol

mm = dff.ix[12823029]['ROMol']
mm

fixRing(mm)

for i, p, m in zip(dff.index, dff['Patent ID'], dff['ROMol']):
    if p == 'EP-2295436-A1':
        dff.ix[i,'ROMol'] = fixRing(m)

dff['InChI'] = dff['ROMol'].map(Chem.MolToInchi)
dff['InChIKey'] = dff['InChI'].map(Chem.InchiToInchiKey)
dff = dff.drop_duplicates('InChIKey')

dff.shape

dff['RD_SMILES'] = dff['ROMol'].map(Chem.MolToSmiles)
dff['base64rdsmi'] = dff['RD_SMILES'].map(base64.b64encode)

fps = [Chem.GetMorganFingerprintAsBitVect(m,2,nBits=2048) for m in dff['ROMol']]
dist_mat = squareform(pdist(fps,'jaccard'))
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=3, n_jobs = 2)
results = mds.fit(dist_mat)
coords = results.embedding_
dff['X'] = [c[0] for c in coords]
dff['Y'] = [c[1] for c in coords]

fig, ax = plt.subplots()
fig.set_size_inches(14.0, 12.0)
#colors = cycle('bgrcmykwbgrcmykbgrcmykwbgrcmykw')
colors = cycle(cm.Dark2(np.linspace(0,1,len(dff_counts.index))))
for name, group in dff.groupby('Patent ID'):
    labels = []
    for i in group.index:
        zz = group.ix[[i],['Patent ID','SCHEMBL ID','base64rdsmi']]
        zz['mol'] = zz['base64rdsmi'].map(lambda x: '<img src="http://{0}/utils/smiles2image/{1}?size=300">'.format(base_url,x))
        del zz['base64rdsmi']
        label = zz.T
        del zz
        label.columns = ['Row: {}'.format(i)]
        labels.append(str(label.to_html()))
    points = ax.scatter(group['X'], group['Y'],c=colors.next(), s=80, alpha=0.8)
    tooltip = mpld3.plugins.PointHTMLTooltip(points, labels, voffset=10, hoffset=10, css=csss)
    mpld3.plugins.connect(fig,tooltip)  

from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
def moltosvg(mol,molSize=(450,350),kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    
    rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    # the MolDraw2D code is not very good at the moment at dealing with atom queries,
    # this is a workaround until that's fixed.
    # The rendering is still not going to be perfect because query bonds are not properly indicated
    opts = drawer.drawOptions()
    for atom in mc.GetAtoms():
        if atom.HasQuery() and atom.DescribeQuery().find('AtomAtomicNum')!=0:
            opts.atomLabels[atom.GetIdx()]=atom.GetSmarts()
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:','')

def MCS_Report(ms,atomCompare=MCS.AtomCompare.CompareAny,**kwargs):
    mcs = MCS.FindMCS(ms,atomCompare=atomCompare,completeRingsOnly=True,timeout=120,**kwargs)
    nAts = np.array([x.GetNumAtoms() for x in ms])
    print 'Mean nAts: {0}, mcs nAts: {1}'.format(nAts.mean(),mcs.numAtoms)
    print 'MCS SMARTS: {0}'.format(mcs.smartsString)
    mcsM = Chem.MolFromSmarts(mcs.smartsString)
    
    mcsM.UpdatePropertyCache(False)
    Chem.SetHybridization(mcsM)
    Chem.Compute2DCoords(mcsM)
    smi = Chem.MolToSmiles(mcsM,isomericSmiles=True)
    print "MCS SMILES: {0}".format(smi)
    img=Draw.MolToImage(mcsM,kekulize=False)
    return mcs.smartsString,smi,img,mcsM

PandasTools.AddMurckoToFrame(dff)

PandasTools.AddMoleculeColumnToFrame(dff,smilesCol='Murcko_SMILES', molCol='MurMol')

PandasTools.AlignToScaffold(dff)

dff[['ROMol','MurMol']].head()

mols = list(dff.MurMol)
smarts,smi,img,mcsM = MCS_Report(mols,threshold=0.8,ringMatchesRingOnly=True)
SVG(moltosvg(mcsM))

core = Chem.MolFromSmarts(smarts)    

pind = []
rgroups = []
for i, m in zip(dff.index, dff.ROMol):
    rgrps = Chem.ReplaceCore(m,core,labelByIndex=True)
    if rgrps:
        s = Chem.MolToSmiles(rgrps, True)
        for e in s.split("."):
            pind.append(e[1:4].replace('*','').replace(']',''))
            rgroups.append(e)
        #print "R-Groups: ", i, s

pind = sorted(list(set(pind)),key=int)

pind = map(int,pind)

Draw.MolToImage(core,includeAtomNumbers=True,highlightAtoms=pind)

HTML('<iframe src=https://dl.dropboxusercontent.com/u/22273283/US6566360B1.pdf width=1000 height=500></iframe>')

sim_mat = []
CUTOFF = 0.8
for i,fp in enumerate(fps):
    tt = DataStructs.BulkTanimotoSimilarity(fps[i],fps)
    for i,t in enumerate(tt):
        if tt[i] < CUTOFF:
            tt[i] = None
    sim_mat.append(tt)
sim_mat=numpy.array(sim_mat) 

sim_df = pd.DataFrame(sim_mat, columns = dff['Chemical ID'], index=dff['Chemical ID'])

sim_df.head()

d = sim_df.count().to_dict()
nn_df = pd.DataFrame(d.items(),columns=['Chemical ID', '# NNs'])
nn_df.set_index('Chemical ID',inplace=True)
nn_df.head()

pd.merge(dff[['Chemical ID','ROMol']],nn_df,left_index=True, right_index=True, sort=False).sort('# NNs',ascending=False).head()

from collections import Counter
rgroups = [r.replace('[9*]','[5*]').replace('[8*]','[6*]') for r in rgroups]
c = Counter(rgroups)

rgroup_counts = pd.DataFrame(c.most_common(20),columns=['RGroups','Frequency'])
rgroup_counts.head(10)

Draw.MolToImage(core,includeAtomNumbers=True,highlightAtoms=pind)

PandasTools.AddMoleculeColumnToFrame(rgroup_counts, smilesCol = 'RGroups')
rgroup_counts.head(10)

rgroup_counts[[r.startswith('[6*]') for r in rgroup_counts.RGroups]].head(10)

dff.ix[dff.InChIKey == 'SECKRCOLJRRGGV-UHFFFAOYSA-N',['SCHEMBL ID','ROMol']]

