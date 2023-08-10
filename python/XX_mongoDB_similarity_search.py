get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')
from IPython.display import Image
from IPython.html.widgets import FloatProgress
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import random
import time
import sys
import os
from itertools import groupby

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append('/home/chembl/ipynb_workbench')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
import chembl_migration_model
from chembl_migration_model.models import *
from chembl_core_model.models import CompoundMols

import pymongo
from bson.binary import Binary
from bson.objectid import ObjectId
from pymongo import MongoClient

client = MongoClient()
db = client.similarity

smiles = MoleculeDictionary.objects.values_list('compoundstructures__canonical_smiles', 'chembl_id')
smi_count = smiles.count()

if 'molecules' not in db.collection_names():
    print 'populating mongodb collection with compounds from chembl...'
    sys.stdout.flush()
    molecules = db.molecules
    percentile = int(smi_count / 100)
    pbar = FloatProgress(min=0, max=smi_count)
    display(pbar)
    chunk_size = 100
    chunk = []
    for i, item in enumerate(smiles):
        if not (i % percentile):
            pbar.value = (i + 1)
        try:
            rdmol = Chem.MolFromSmiles(item[0])
        except:
            continue
        if not rdmol:
            continue
        mol_data = {
            'smiles': Chem.MolToSmiles(rdmol, isomericSmiles=True),
            'chembl_id': item[1],
            'rdmol': Binary(rdmol.ToBinary()),
        }
        chunk.append(mol_data)
        if len(chunk) == chunk_size:
            molecules.insert_many(chunk)
            chunk = []
    molecules.insert_many(chunk)
    chunk = []        
    pbar.value = smi_count
    print '%s molecules loaded successfully' % molecules.count()

print 'precalculating fingerprints...'
sys.stdout.flush()
mol_count = molecules.count()
percentile = int(mol_count / 100)
pbar = FloatProgress(min=0, max=mol_count)
display(pbar)
for i, molecule in enumerate(db.molecules.find()):
    if not (i % percentile):
        pbar.value = (i + 1)
    rdmol = Chem.Mol(molecule['rdmol'])
    mfp = list(AllChem.GetMorganFingerprintAsBitVect(rdmol, 2, nBits=2048).GetOnBits())
    db.molecules.update_one({'_id': molecule['_id']}, {"$set":{"mfp":{'bits': mfp, 'count': len(mfp)}}} )
pbar.value = mol_count    
print '...done'

response = db.molecules.aggregate([{'$group': {'_id': '$mfp.count', 'total': {'$sum': 1}}}])
data = pd.DataFrame().from_dict([r for r in response])
fig = plt.figure()
fig.set_size_inches(18, 6)
plt.ylabel('Number of molecules')
plt.xlabel('Number of 1-bits in fingerprint')
h = plt.hist(data._id, weights=data.total, histtype='bar', bins=105, rwidth=2)

print 'computing fingerprint bit counts...'
sys.stdout.flush()
counts = {}
mol_count = molecules.count()
percentile = int(mol_count / 100)
pbar = FloatProgress(min=0, max=mol_count)
display(pbar)
for molecule in db.molecules.find():
    if not (i % percentile):
        pbar.value = (i + 1)
    for bit in molecule['mfp']['bits']:
        counts[bit] = counts.get(bit, 0) + 1

counts_it = counts.items()
chunk_size = 100
for chunk in [counts_it[i:i + chunk_size] for i in range(0, len(counts_it), chunk_size)]:
    db.mfp_counts.insert_many([{'_id': k, 'count': v} for k,v in chunk])
pbar.value = mol_count    
print '... done'    

db.molecules.create_index('mfp.bits')
db.molecules.create_index('mfp.count')

def similarity_client(smiles, threshold=0.8):
    """Perform a similarity search on the client, with initial screening to improve performance."""
    if not smiles:
        return
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return
    qfp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).GetOnBits())
    qn = len(qfp)                           # Number of bits in query fingerprint
    qmin = int(ceil(qn * threshold))        # Minimum number of bits in results fingerprints
    qmax = int(qn / threshold)              # Maximum number of bits in results fingerprints
    ncommon = qn - qmin + 1                 # Number of fingerprint bits in which at least one must be in common
    # Get list of bits where at least one must be in result fp. Use least popular bits if possible.
    if db.mfp_counts:
        reqbits = [count['_id'] for count in db.mfp_counts.find({'_id': {'$in': qfp}}).sort('count', 1).limit(ncommon)]
    else:
        reqbits = qfp[:ncommon]
    results = []
    for fp in db.molecules.find({'mfp.bits': {'$in': reqbits}, 'mfp.count': {'$gte': qmin, '$lte': qmax}}):
        intersection = len(set(qfp) & set(fp['mfp']['bits']))
        pn = fp['mfp']['count']
        tanimoto = float(intersection) / (pn + qn - intersection)
        if tanimoto >= threshold:
            results.append((tanimoto, fp['chembl_id'], fp['smiles']))
    return results

# aspirin
similarity_client('O=C(Oc1ccccc1C(=O)O)C')

def similarity_search_fp(smiles, threshold=0.8):
    """Perform a similarity search using aggregation framework."""
    if not smiles:
        return
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return
    qfp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).GetOnBits())    
    qn = len(qfp)                           # Number of bits in query fingerprint
    qmin = int(ceil(qn * threshold))        # Minimum number of bits in results fingerprints
    qmax = int(qn / threshold)              # Maximum number of bits in results fingerprints
    ncommon = qn - qmin + 1                 # Number of fingerprint bits in which at least 1 must be in common
    if db.mfp_counts:
        reqbits = [count['_id'] for count in db.mfp_counts.find({'_id': {'$in': qfp}}).sort('count', 1).limit(ncommon)]
    else:
        reqbits = qfp[:ncommon]
    aggregate = [
        {'$match': {'mfp.count': {'$gte': qmin, '$lte': qmax}, 'mfp.bits': {'$in': reqbits}}},
        {'$project': {
            'tanimoto': {'$let': {
                'vars': {'common': {'$size': {'$setIntersection': ['$mfp.bits', qfp]}}},
                'in': {'$divide': ['$$common', {'$subtract': [{'$add': [qn, '$mfp.count']}, '$$common']}]}
            }},
        'smiles': 1,
        'chembl_id': 1
        }},
        {'$match': {'tanimoto': {'$gte': threshold}}}
    ]
    response = db.molecules.aggregate(aggregate)
    return [(r['tanimoto'], r['smiles'], r['chembl_id']) for r in response]

similarity_search_fp('O=C(Oc1ccccc1C(=O)O)C')

sample_size = 1000
rand_smpl = [ smiles[i][0] for i in sorted(random.sample(xrange((smiles.count())), sample_size)) ]
repetitions = 5

timings = []
for thresh in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    print 'measuring performance for similarity {0}'.format(thresh)
    sys.stdout.flush()
    rep_times = []
    for i in range(repetitions):
        start = time.time()
        for sample in rand_smpl:
            _ = similarity_search_fp(sample, thresh)
        stop = time.time()
        rep_times.append(stop-start)
    timings.append((thresh, np.mean(rep_times)))

print timings

thresholds = np.array([t[0] for t in reversed(timings)])
times = np.array([t[1] for t in reversed(timings)])
fig = plt.figure()
fig.set_size_inches(6, 6)
xnew = np.linspace(thresholds.min(), thresholds.max(), num=41, endpoint=True)
f2 = interp1d(thresholds, times, kind='quadratic')
plt.plot(thresholds, times, 'o', xnew, f2(xnew), '-')
plt.ylabel('Mean query time (ms)')
plt.xlabel('Similarity threshold')
plt.show()

def get_permutations(len_permutations=2048, num_permutations=100):
    return map(lambda _: np.random.permutation(2048), range(num_permutations))

def get_min_hash(mol, permutations):
    qfp_bits = [int(n) for n in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)]
    min_hash = []
    for perm in permutations:
        for idx, i in enumerate(perm):
            if qfp_bits[i]:
                min_hash.append(idx)
                break            
    return min_hash

def hash_to_buckets(min_hash, num_buckets=25, nBits=2048):
    if len(min_hash) % num_buckets:
        raise Exception('number of buckets must be divisiable by the hash length')
    buckets = []
    hash_per_bucket = int(len(min_hash) / num_buckets)
    num_bits = (nBits-1).bit_length()
    if num_bits * hash_per_bucket > sys.maxint.bit_length():
        raise Exception('numbers are too large to produce valid buckets')
    for b in range(num_buckets):
        buckets.append(reduce(lambda x,y: (x << num_bits) + y, min_hash[b:(b + hash_per_bucket)]))
    return buckets
        

db.permutations.insert_many([{'_id':i, 'permutation': perm.tolist()} for i, perm in enumerate(get_permutations())])

permutations = [p['permutation'] for p in db.permutations.find()]
print 'precalculating locality-sensitive hashing groups...'
sys.stdout.flush()
mol_count = db.molecules.count()
percentile = int(mol_count / 100)
pbar = FloatProgress(min=0, max=mol_count)
display(pbar)
for i, molecule in enumerate(db.molecules.find()):
    if not (i % percentile):
        pbar.value = (i + 1)
    rdmol = Chem.Mol(molecule['rdmol'])
    min_hash = get_min_hash(rdmol, permutations)
    hash_groups = hash_to_buckets(min_hash)
    db.molecules.update_one({'_id': molecule['_id']}, {"$set":{"lsh" : hash_groups}} )
pbar.value = mol_count    
print '...done'

print 'constructing hash maps...'
sys.stdout.flush()
mol_count = db.molecules.count()
percentile = int(mol_count / 100)
pbar = FloatProgress(min=0, max=mol_count)
display(pbar)
for i, molecule in enumerate(db.molecules.find()):
    if not (i % percentile):
        pbar.value = (i + 1)
    hash_groups = molecule["lsh"]
    for n_hash, lsh_hash in enumerate(hash_groups):
        db['hash_' + str(n_hash)].update_one({'_id':lsh_hash}, {'$push':{'molecules': molecule['_id']}}, True)
pbar.value = mol_count
print '...done'

permutations = [p['permutation'] for p in db.permutations.find()]

def similarity_search_lsh(smiles, threshold=0.8):
    """Perform a similarity search using aggregation framework."""
    if not smiles:
        return
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    qfp = list(fp.GetOnBits())
    qfp_bits = [int(n) for n in fp]
    min_hash = []
    for perm in permutations:
        for idx, i in enumerate(perm):
            if qfp_bits[i]:
                min_hash.append(idx)
                break
    hash_groups = hash_to_buckets(min_hash)    
    qn = len(qfp)                           # Number of bits in query fingerprint
    qmin = int(ceil(qn * threshold))        # Minimum number of bits in results fingerprints
    qmax = int(qn / threshold)              # Maximum number of bits in results fingerprints
    ncommon = qn - qmin + 1                 # Number of fingerprint bits in which at least 1 must be in common
    if db.mfp_counts:
        reqbits = [count['_id'] for count in db.mfp_counts.find({'_id': {'$in': qfp}}).sort('count', 1).limit(ncommon)]
    else:
        reqbits = qfp[:ncommon]
    
    nested_res = [ list(i)[0]['molecules'] for i in 
            [db['hash_' + str(i)].find({'_id':h},{'molecules':1}) for i,h in enumerate(hash_groups)]]
    
    hashed_ids = [ObjectId(x) for x in (set([str(item) for sublist in nested_res for item in sublist]))]
    aggregate = [
        {'$match': {'_id':{'$in': hashed_ids}, 'mfp.count': {'$gte': qmin, '$lte': qmax}, 'mfp.bits': {'$in': reqbits}}},
        {'$project':{            
         'tanimoto': {'$let': {
                'vars': {'common': {'$size': {'$setIntersection': ['$mfp.bits', qfp]}}},
                'in': {'$divide': ['$$common', {'$subtract': [{'$add': [qn, '$mfp.count']}, '$$common']}]}
            }},
        'smiles': 1,
        'chembl_id': 1}},
        {'$match': {'tanimoto': {'$gte': threshold}}},
    ]
    response = db.molecules.aggregate(aggregate)
    return [(r['tanimoto'], r['smiles'], r['chembl_id']) for r in response]

similarity_search_lsh('O=C(Oc1ccccc1C(=O)O)C')

timings_lsh = []
for thresh in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]: #[0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    print 'measuring performance for similarity {0}'.format(thresh)
    sys.stdout.flush()
    rep_times = []
    for i in range(repetitions):
        start = time.time()
        for sample in rand_smpl:
            _ = similarity_search_lsh(sample, thresh)
        stop = time.time()
        rep_times.append(stop-start)
    timings_lsh.append((thresh, np.mean(rep_times)))

timings_lsh

thresholds = np.array([t[0] for t in reversed(timings_lsh)])
times = np.array([t[1] for t in reversed(timings_lsh)])
print timings_lsh
print times
fig = plt.figure()
fig.set_size_inches(6, 6)
xnew = np.linspace(thresholds.min(), thresholds.max(), num=41, endpoint=True)
f2 = interp1d(thresholds, times, kind='slinear')
plt.plot(thresholds, times, 'o', xnew, f2(xnew), '-')
plt.ylabel('Median query time (ms)')
plt.xlabel('Similarity threshold')
plt.show()

discrepancies = []
discrepancies_details = []
for thresh in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]: #[0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    print 'measuring discrepancies for similarity {0}'.format(thresh)
    sys.stdout.flush()
    rep_discrepancies = []
    rep_coherent = []
    rep_err_compounds = []
    for i in range(1):
        dis_sum = 0
        sum_sum = 0
        err_comp_sum = 0
        for sample in rand_smpl:
            sim_lsh = similarity_search_lsh(sample, thresh)
            sim_fp = similarity_search_fp(sample, thresh)
            lsh_res = set([x[2] for x in sim_lsh if (x and len(x) > 2)]) if sim_lsh else set()
            reg_res = set([x[2] for x in sim_fp if (x and len(x) > 2)]) if sim_fp else set()
            difference = (lsh_res^reg_res)
            if len(difference):
                err_comp_sum += 1
            dis_sum += len(difference)
            sum_sum += len(lsh_res|reg_res)
            discrepancies_details.append((sample, thresh, difference))
        rep_discrepancies.append(dis_sum)
        rep_coherent.append(sum_sum)
        rep_err_compounds.append(err_comp_sum)
    discrepancies.append((thresh, np.mean(rep_discrepancies), np.mean(rep_coherent), np.mean(rep_err_compounds)))

print discrepancies

thresholds = np.array([t[0] for t in reversed(discrepancies)])
errors = np.array([t[1] / t[2] for t in reversed(discrepancies)])
fig = plt.figure()
fig.set_size_inches(6, 6)
xnew = np.linspace(thresholds.min(), thresholds.max(), num=41, endpoint=True)
f2 = interp1d(thresholds, errors, kind='quadratic')
plt.plot(thresholds, errors, 'o', xnew, f2(xnew), '-')
plt.ylabel('Percent of discrepancies')
plt.xlabel('Similarity threshold')
plt.show()

thresholds = np.array([t[0] for t in reversed(discrepancies)])
errors = np.array([t[3] / 10.0 for t in reversed(discrepancies)])
fig = plt.figure()
fig.set_size_inches(6, 6)
xnew = np.linspace(thresholds.min(), thresholds.max(), num=41, endpoint=True)
f2 = interp1d(thresholds, errors, kind='quadratic')
plt.plot(thresholds, errors, 'o', xnew, f2(xnew), '-')
plt.ylabel('Percent of discrepancies')
plt.xlabel('Similarity threshold')
plt.show()

thresholds = np.array([t[0] for t in reversed(timings)])
times = np.array([t[1] for t in reversed(timings)])
fig = plt.figure()
fig.set_size_inches(6, 6)
xnew = np.linspace(thresholds.min(), thresholds.max(), num=41, endpoint=True)
f2 = interp1d(thresholds, times, kind='cubic')
plt.plot(thresholds, times, 'o', xnew, f2(xnew), '-')

thresholds_lsh = np.array([t[0] for t in reversed(timings_lsh)])
times_lsh = np.array([t[1] for t in reversed(timings_lsh)])
xnew_lsh = np.linspace(thresholds_lsh.min(), thresholds_lsh.max(), num=41, endpoint=True)
f3 = interp1d(thresholds_lsh, times_lsh, kind='cubic')

_, swain = plt.plot(thresholds, times, 'o', xnew, f2(xnew), '-')
_, lsh = plt.plot(thresholds_lsh, times_lsh, 'o', xnew_lsh, f3(xnew_lsh), '--')
plt.legend([swain, lsh], ['Swain', 'LSH'])
plt.ylabel('Median query time (ms)')
plt.xlabel('Similarity threshold')
plt.show()

repetitions = 1
timings_cartrdge = []
for thresh in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    print 'measuring performance for similarity {0}'.format(thresh)
    sys.stdout.flush()
    rep_times = []
    for i in range(repetitions):
        start = time.time()
        for sample in rand_smpl:
            _ = len(CompoundMols.objects.similar_to(sample, int(thresh * 100)).values_list('molecule_id', 'similarity'))
        stop = time.time()
        rep_times.append(stop-start)
    timings_cartrdge.append((thresh, np.mean(rep_times)))

timings_cartrdge

thresholds = np.array([t[0] for t in reversed(timings)])
times = np.array([t[1] for t in reversed(timings)])
fig = plt.figure()
fig.set_size_inches(8, 8)
xnew = np.linspace(thresholds.min(), thresholds.max(), num=41, endpoint=True)
f2 = interp1d(thresholds, times, kind='quadratic')
plt.plot(thresholds, times, 'o', xnew, f2(xnew), '-')

thresholds_lsh = np.array([t[0] for t in reversed(timings_lsh)])
times_lsh = np.array([t[1] for t in reversed(timings_lsh)])
xnew_lsh = np.linspace(thresholds_lsh.min(), thresholds_lsh.max(), num=41, endpoint=True)
f3 = interp1d(thresholds_lsh, times_lsh, kind='quadratic')

thresholds_car = np.array([t[0] for t in reversed(timings_cartrdge)])
times_car = np.array([t[1] for t in reversed(timings_cartrdge)])
xnew_car = np.linspace(thresholds_car.min(), thresholds_car.max(), num=41, endpoint=True)
f4 = interp1d(thresholds_car, times_car, kind='quadratic')

_, swain = plt.plot(thresholds, times, 'o', xnew, f2(xnew), '-')
_, lsh = plt.plot(thresholds_lsh, times_lsh, 'o', xnew_lsh, f3(xnew_lsh))
_, cartridge = plt.plot(thresholds_car, times_car, 'o', xnew_car, f4(xnew_car))
plt.ylabel('Mean query time (ms)')
plt.xlabel('Similarity threshold')
plt.legend([swain, lsh, cartridge], ['Swain', 'LSH', 'postgres'])
plt.show()



