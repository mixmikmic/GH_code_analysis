import os
import requests
import numpy as np
import pandas as pd
import tables  # Required by h5py to write a pandas table
import h5py

get_ipython().run_cell_magic('time', '', '"""\nDownload expression data from xena and save in an hdf5 file. This can take around \n30 minutes between the download and the conversion from tsv into float32 dataframes\nWe download manually vs. passing read_csv a url directly as the latter times\nout with this size file. Note we invert the expression matrix to conform \nto machine learning where rows are samples vs. gene expression files where \nrows are features (gene, transcript, or variant) and columns are \ninstances (sample or single cell)\n"""\nif not os.path.exists("data"):\n    os.makedirs("data")\n    \nif not os.path.exists("data/TcgaTargetGtex_rsem_gene_tpm.tsv.gz"):\n    print("Downloading TCGA, TARGET and GTEX expression data from UCSC Xena")\n    r = requests.get("https://toil.xenahubs.net/download/TcgaTargetGtex_rsem_gene_tpm.gz")\n    r.raise_for_status()\n    with open("data/TcgaTargetGtex_rsem_gene_tpm.tsv.gz", "wb") as f:\n        for chunk in r.iter_content(32768):\n            f.write(chunk)\n\nif not os.path.exists("data/TcgaTargetGtex_rsem_gene_tpm.hd5"):\n    print("Converting expression to dataframe and storing in hdf5 file")\n    expression = pd.read_csv("data/TcgaTargetGtex_rsem_gene_tpm.tsv.gz", \n                             sep="\\t", index_col=0).dropna().astype(np.float32).T\n    expression.to_hdf("data/TcgaTargetGtex_rsem_gene_tpm.hd5", "expression", mode="w", format="fixed")\n\nX = pd.read_hdf("data/TcgaTargetGtex_rsem_gene_tpm.hd5", "expression").sort_index(axis=0)\nprint("X: samples={} genes={}".format(*X.shape))')

X.head()

# Read in the sample labels from Xena ie clinical/phenotype information on each sample
Y = pd.read_table("https://toil.xenahubs.net/download/TcgaTargetGTEX_phenotype.txt.gz", compression="gzip",
                  header=0, names=["id", "category", "disease", "primary_site", "sample_type", "gender", "study"],
                  sep="\t", encoding="ISO-8859-1", index_col=0, dtype="str").sort_index(axis=0)

# Compute and add a tumor/normal column - TCGA and TARGET have some normal samples, GTEX is all normal.
Y["tumor_normal"] = Y.apply(
    lambda row: "Normal" if row["sample_type"] in ["Cell Line", "Normal Tissue", "Solid Tissue Normal"]
    else "Tumor", axis=1)

Y[0:100:4000].head()

Y.describe()

# Use the tissue location as the class label for the purposes of strattification
class_attribute = "primary_site"

# Tumor vs. Normal is the binary attribute we'll use to train on
label_attribute = "tumor_normal"

# Remove rows where the class is null or the sample is missing
Y_not_null = Y[pd.notnull(Y[class_attribute])]
intersection = X.index.intersection(Y_not_null.index)
X_clean = X[X.index.isin(intersection)]
Y_clean = Y[Y.index.isin(intersection)]

# Make sure the label and example samples are in the same order
assert(X_clean.index.equals(Y_clean.index))

print(intersection.shape[0], "samples with non-null labels")

# Convert tumor/normal labels to binary 1/0
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_binary = encoder.fit_transform(Y_clean["tumor_normal"])

# Convert classes into numbers
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(Y_clean[class_attribute].values)
classes = encoder.transform(Y_clean[class_attribute])
print("Total classes for stratification:", len(set(classes)))
class_labels = encoder.classes_

get_ipython().run_cell_magic('time', '', '# Split into stratified training and test sets based on classes (i.e. tissue type)\nfrom sklearn.model_selection import StratifiedShuffleSplit\nsplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\nfor train_index, test_index in split.split(X_clean.values, Y_clean[class_attribute]):\n    X_train, X_test = X.values[train_index], X_clean.values[test_index]\n    y_train, y_test = y_binary[train_index], y_binary[test_index]\n    classes_train, classes_test = classes[train_index], classes[test_index]')

"""
Feature labels are ensemble ids, convert to hugo gene names for use in interpreting
hidden layers in any trained models as they are better known to most bioinformaticians 
and clinicians. We're using an assembled table from John Vivian @ UCSC here. Another
option would be ftp://ftp.ebi.ac.uk/pub/databases/genenames/new/tsv/hgnc_complete_set.txt
"""
ensemble_to_hugo = pd.read_table(
    "https://github.com/jvivian/docker_tools/blob/master/gencode_hugo_mapping/attrs.tsv?raw=true",
    index_col=0)
ensemble_to_hugo.head()

# The ensembl to hugo table has duplicates due to the many transcripts that map
# to a gene. Remove the duplicates and then lookup hugo for each ensemble id
hugo = ensemble_to_hugo[~ensemble_to_hugo.index.duplicated(keep='first')].loc[X_clean.columns.values]["geneName"].fillna("")

# Make sure we end up with the order of features being identical as some ensemble id's
# map to the same hugo gene id
assert(X_clean.columns.equals(hugo.index))

"""
Write to an h5 file for training (see above for details on each dataset)
"""
with h5py.File("data/tumor_normal.h5", "w") as f:
    f.create_dataset('X_train', X_train.shape, dtype='f')[:] = X_train
    f.create_dataset('X_test', X_test.shape, dtype='f')[:] = X_test
    f.create_dataset('y_train', y_train.shape, dtype='i')[:] = y_train
    f.create_dataset('y_test', y_test.shape, dtype='i')[:] = y_test
    f.create_dataset('classes_train', y_train.shape, dtype='i')[:] = classes_train
    f.create_dataset('classes_test', y_test.shape, dtype='i')[:] = classes_test
    f.create_dataset('features', X_clean.columns.shape, 'S10', 
                     [l.encode("ascii", "ignore") for l in X_clean.columns.values])
    f.create_dataset('genes', hugo.shape, 'S10', 
                     [l.encode("ascii", "ignore") for l in hugo.values.tolist()])
    f.create_dataset('labels', (2, 1), 'S10', 
                     [l.encode("ascii", "ignore") for l in ["Normal", "Tumor"]])
    f.create_dataset('class_labels', (len(class_labels), 1), 'S10', 
                     [l.encode("ascii", "ignore") for l in class_labels])

