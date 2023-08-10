import pandas as pd

train_queries = pd.read_csv('../nfcorpus/train.all.queries', sep='\t', header=None)
train_queries.columns = ['id', 'text']
dev_queries = pd.read_csv('../nfcorpus/dev.all.queries', sep='\t', header=None)
dev_queries.columns = ['id', 'text']
test_queries = pd.read_csv('../nfcorpus/test.all.queries', sep='\t', header=None)
test_queries.columns = ['id', 'text']

train_matrix = pd.read_pickle('pickle/train_matrix.pkl')
dev_matrix = pd.read_pickle('pickle/dev_matrix.pkl')
test_matrix = pd.read_pickle('pickle/test_matrix.pkl')

#We also want to get some information about our given relevance scores
train_rel = pd.read_csv('../nfcorpus/train.2-1-0.qrel', sep='\t', header=None)
test_rel = pd.read_csv('../nfcorpus/test.2-1-0.qrel', sep='\t', header=None)
dev_rel = pd.read_csv('../nfcorpus/dev.2-1-0.qrel', sep='\t', header=None)
#column 1 is always 0, so drop it
train_rel = train_rel.drop([1], axis=1)
dev_rel = dev_rel.drop([1], axis=1)
test_rel = test_rel.drop([1], axis=1)
train_rel.columns = ['qid', 'docid', 'rel']
dev_rel.columns = ['qid', 'docid', 'rel']
test_rel.columns = ['qid', 'docid', 'rel']

all_queries = pd.concat([train_queries, dev_queries, test_queries])
all_matrices = pd.concat([train_matrix, dev_matrix, test_matrix])
all_rels = pd.concat([train_rel, dev_rel, test_rel])

train_queries.describe()

train_matrix.sum(axis=1).describe()

dev_queries.describe()

dev_matrix.sum(axis=1).describe()

test_queries.describe()

test_matrix.sum(axis=1).describe()

all_queries.describe()

all_matrices.sum(axis=1).describe()

#How many uniquely relevant documents do we have (or why does raw contain 5000 docs, and train/dev/test together only 3633)

docids_train = train_rel.docid
docids_dev = dev_rel.docid
docids_test = test_rel.docid
docids = pd.concat([docids_train, docids_dev, docids_test])
len(docids.unique())
#When creating the dataset, the researchers from HD only kept those documents, that were at least once relevant for a query
#This means, roughly 1700 documents from the raw corpus were never relevant, and therefore excluded

#How many relevance scores do we have in total, avg per query over all sets
all_rels.describe()

len(all_rels)/len(all_queries)



