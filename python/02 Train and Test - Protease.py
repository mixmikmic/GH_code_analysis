from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from sklearn.cross_validation import train_test_split, cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.preprocessing import LabelBinarizer

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import custom_funcs as cf

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

# Read in the protease inhibitor data
data = pd.read_csv('drug_data/hiv-protease-data.csv', index_col='SeqID')
drug_cols = data.columns[0:8]
feat_cols = data.columns[8:]

# Read in the consensus data
consensus = SeqIO.read('sequences/hiv-protease-consensus.fasta', 'fasta')

consensus_map = {i:letter for i, letter in enumerate(str(consensus.seq))}

# Because there are '-' characters in the dataset, representing consensus sequence at each of the positions, 
# they need to be replaced with the actual consensus letter.

for i, col in enumerate(feat_cols):
    # Replace '-' with the consensus letter.
    data[col] = data[col].replace({'-':consensus_map[i]})
    
    # Replace '.' with np.nan
    data[col] = data[col].replace({'.':np.nan})
    
    # Replace 'X' with np.nan
    data[col] = data[col].replace({'X':np.nan})
    
# Drop any feat_cols that have np.nan inside them. We don't want low quality sequences.
data.dropna(inplace=True, subset=feat_cols)
data

# Grab out the drug name to be tested.
DRUG = drug_cols[0]

# I have written a custom function that takes in the data, and reduces it such that we only consider one drug column
# value, and the corresponding amino acid sequences. The NaN values are also dropped, as most machine learning
# algorithms in scikit-learn cannot deal with NaN values. Finally, the data are log10 transformed.
X, Y = cf.split_data_xy(data, feat_cols, DRUG)

# Binarize the sequence features such that there are 99 x 20 columns in total.
# The purpose of binarizing is to turn the string labels into numeric labels. The most naive way to do this is to 
# take every amino acid position in the sequence alignment, and turn it into 20 columns of 1s and 0s corresponding 
# to whether a particular amino acid is present or not.

lb = LabelBinarizer()
lb.fit(list('CHIMSVAGLPTRFYWDNEQK'))

X_binarized = pd.DataFrame()

for col in X.columns:
    binarized_cols = lb.transform(X[col])
    
    for i, c in enumerate(lb.classes_):
        X_binarized[col + '_' + c] = binarized_cols[:,i]
X_binarized

# Next step is to split the data into a training set and a test set. 
# We use the train_test_split function provided by the scikit-learn package to do this.
tts_data = X_train, X_test, Y_train, Y_test = train_test_split(X_binarized, Y, test_size=0.33)

# Train a bunch of ensemble regressors:

## Random Forest
kwargs = {'n_jobs':-1, 'n_estimators':1000}
rfr, rfr_preds, rfr_mse, rfr_r2 = cf.train_model(*tts_data, model=RandomForestRegressor, modelargs=kwargs)

## Gradient Boosting
kwargs = {'n_estimators':1000}
gbr, gbr_preds, gbr_mse, gbr_r2 = cf.train_model(*tts_data, model=GradientBoostingRegressor, modelargs=kwargs)

## AdaBoost
kwargs = {'n_estimators':1000}
abr, abr_preds, abr_mse, abr_r2 = cf.train_model(*tts_data, model=AdaBoostRegressor, modelargs=kwargs)

## ExtraTrees
kwargs = {'n_estimators':1000, 'n_jobs':-1}
etr, etr_preds, etr_mse, etr_r2 = cf.train_model(*tts_data, model=ExtraTreesRegressor, modelargs=kwargs)

## Bagging
bgr, bgr_preds, bgr_mse, bgr_r2 = cf.train_model(*tts_data, model=BaggingRegressor)

# Compare the trained models. Which one minimizes the mean squared error the most?

rfr_mse, gbr_mse, abr_mse, etr_mse, bgr_mse

# Qualitatively, what is the math behind the MSE score? Are you looking to minimize or maximize it?

models = dict()
models['rfr'] = RandomForestRegressor(n_estimators=100, n_jobs=-1)
models['gbr'] = GradientBoostingRegressor(n_estimators=100)
models['abr'] = AdaBoostRegressor(n_estimators=100)
models['etr'] = ExtraTreesRegressor(n_estimators=100, n_jobs=-1)
models['bgr'] = BaggingRegressor()

scores = cross_val_score(models['gbr'], X_binarized, Y, cv=ShuffleSplit(n=len(Y)))
scores

# Using the coding pattern illustrated above, tweak the parameters as documented in the scikit-learn documentation.
# Can you improve the model predictions? 
"""
Basically, the idea is to tweak the parameters.
"""

# Pick the model that gives the best MSE score (minimize MSE score), and use it to make predictions.

# Read in the sequence data for sequences that do not have predictions made.
proteases_alignment = [s for s in SeqIO.parse('sequences/proteases_downsampled.fasta', 'fasta') if len(s) == len(consensus.seq)]
proteases_alignment = MultipleSeqAlignment(proteases_alignment)
proteases_alignment

# Binarize the mutliple sequence alignment.
proteases_df = pd.DataFrame()

for col in range(proteases_alignment.get_alignment_length()):
    binarized_cols = lb.transform([k for k in proteases_alignment[:,col]])
    
    for i, c in enumerate(lb.classes_):
        proteases_df[str(col) + '_' + c] = binarized_cols[:,i]
        
# Add in the index.
index = []
for s in proteases_alignment:
    index.append(s.id)
    
proteases_df.index = index

proteases_df

# Make predictions using the best model - ExtraTrees Regressor
etr_preds = pd.DataFrame(etr.predict(proteases_df))
etr_preds['traits'] = proteases_df.index
etr_preds.columns = ['{0}_resistance'.format(DRUG), 'traits']

etr_preds.set_index('traits', inplace=True)

pd.DataFrame(etr_preds).to_csv('csv/{0}_etr_preds.tsv'.format(DRUG), sep='\t')



