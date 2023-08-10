get_ipython().magic('pylab inline')
from IPython.display import Image

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw

import pandas as pd
from pandas import concat

from collections import OrderedDict

# By default, the API connects to the main ChEMBL database; set it to use the local version (i.e. myChEMBL) instead...
from chembl_webresource_client.settings import Settings

Settings.Instance().NEW_CLIENT_URL = 'http://localhost/chemblws'

from chembl_webresource_client.new_client import new_client

from sklearn.externals import joblib

rcParams['figure.figsize'] = 10,10

def calc_scores(classes):
    p = []
    for c in classes:
        p.append(pred_score(c))
    return p

def pred_score(trgt):
    diff = morgan_nb.estimators_[classes.index(trgt)].feature_log_prob_[1] - morgan_nb.estimators_[classes.index(trgt)].feature_log_prob_[0]
    return sum(diff*fp)

morgan_nb = joblib.load('/home/chembl/models_21/10uM/mNB_10uM_all.pkl')

classes = list(morgan_nb.targets)

len(classes)

import warnings
warnings.filterwarnings('ignore') 
warnings.filterwarnings("ignore", category=DeprecationWarning)

smiles = 'O[C@@H](CNCCCC#CC1=CC=C(C=C1)NC(=O)C=1C=C(C=CC1)S(=O)(=O)C=1C=C2C(=C(C=NC2=C(C1)C)C(=O)N)NC1=CC(=CC=C1)OC)C1=C2C=CC(NC2=C(C=C1)O)=O'

mol = Chem.MolFromSmiles(smiles)

mol

info={}
fp = Chem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048, bitInfo=info)

predictions = pd.DataFrame(zip(classes, calc_scores(classes),list(morgan_nb.predict_proba(fp)[0])),columns=['id','score','proba'])

predictions.head()

predictions['proba'].hist()

predictions['score'].hist()

top_preds = predictions.sort(columns=['proba'],ascending=False).head(10)

top_preds

def fetch_WS(trgt):
    targets = new_client.target
    return targets.get(trgt)

plist = []
for i,e in enumerate(top_preds['id']):
    p = pd.DataFrame(fetch_WS(e), index=(i,))
    plist.append(p)
target_info = concat(plist)

target_info.shape

target_info

result = pd.merge(top_preds, target_info, left_on='id', right_on='target_chembl_id')

result

bit_scores = (morgan_nb.estimators_[classes.index(result['id'][1])].feature_log_prob_[1] - morgan_nb.estimators_[classes.index(result['id'][1])].feature_log_prob_[0])*fp

frags = OrderedDict()
for k in info.keys():
    if bit_scores[k] > 0.1:
        atomId,radius = info[k][0]
        env=Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atomId)
        ats = set([atomId])
        for bidx in env:
            bond = mol.GetBondWithIdx(bidx)
            ats.add(bond.GetBeginAtomIdx())
            ats.add(bond.GetEndAtomIdx())
            frag = Chem.MolFragmentToSmiles(mol,atomsToUse=list(ats),bondsToUse=env,rootedAtAtom=atomId)
            legend = str(round(bit_scores[k],2))
            frags[k] = (legend,frag)

legends = [l[1][0] for l in sorted(frags.items(), key=lambda t: t[1][0], reverse=True)][:10]
ffrags = [l[1][1] for l in sorted(frags.items(), key=lambda t: t[1][0], reverse=True)][:10]

fmols=[Chem.MolFromSmarts(s) for s in ffrags]

mol

Draw.MolsToGridImage(fmols, molsPerRow=5, legends=legends, useSVG=False)

