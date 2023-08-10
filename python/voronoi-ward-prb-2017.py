get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matminer.datasets import dataframe_loader
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital, IonProperty
from matminer.featurizers.structure import SiteStatsFingerprint, StructuralHeterogeneity, ChemicalOrdering, StructureComposition 
from matminer.featurizers.structure import MaximumPackingEfficiency
from matminer.utils.conversions import dict_to_object
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from scipy import stats
from tqdm import tqdm_notebook as tqdm
import numpy as np

featurizer = MultipleFeaturizer([
    SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
    StructuralHeterogeneity(),
    ChemicalOrdering(),
    MaximumPackingEfficiency(),
    SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
    StructureComposition(Stoichiometry()),
    StructureComposition(ElementProperty.from_preset("magpie")),
    StructureComposition(ValenceOrbital(props=['frac'])),
    StructureComposition(IonProperty(fast=True))
])

get_ipython().run_cell_magic('time', '', "data = dataframe_loader.load_flla()\nprint('Loaded {} entries'.format(len(data)))")

data['structure'] = dict_to_object(data['structure'])

get_ipython().run_cell_magic('time', '', "print('Total number of features:', len(featurizer.featurize(data['structure'][0])))\nprint('Number of sites in structure:', len(data['structure'][0]))")

get_ipython().run_cell_magic('time', '', "X = featurizer.featurize_many(data['structure'], ignore_errors=True)")

X = np.array(X)
print('Input data shape:', X.shape)

failed = np.any(np.isnan(X), axis=1)
print('Number failed: {}/{}'.format(np.sum(failed), len(failed)))

model = Pipeline([
    ('imputer', Imputer()), # For the failed structures
    ('model', RandomForestRegressor(n_estimators=150, n_jobs=-1))
])

get_ipython().run_cell_magic('time', '', "model.fit(X, data['formation_energy_per_atom'])")

maes = []
for train_ids, test_ids in tqdm(ShuffleSplit(train_size=3000, n_splits=20).split(X)):
    # Split off the datasets
    train_X = X[train_ids, :]
    train_y = data['formation_energy_per_atom'].iloc[train_ids]
    test_X = X[test_ids, :]
    test_y = data['formation_energy_per_atom'].iloc[test_ids]
    
    # Train the model
    model.fit(train_X, train_y)
    
    # Run the model, compute MAE
    predict_y = model.predict(test_X)
    maes.append(np.abs(test_y - predict_y).mean())

print('MAE: {:.3f}+/-{:.3f} eV/atom'.format(np.mean(maes), stats.sem(maes)))



