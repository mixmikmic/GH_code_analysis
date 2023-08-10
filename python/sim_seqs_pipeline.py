import msprime
import numpy as np
import dendropy
from dendropy.interop import seqgen
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import time

ml_feats = np.empty(shape = [0,256]).astype(int) # holds our data
ml_labels = np.empty(shape = [0,1]) # holds our labels

samplenum = 2000 # how many simulation samples
# make list of migration rates
mig_rate = np.append(np.random.gamma(shape = 1,  # make half of the rates positive and the other half zero
                                     scale=.00001,
                                     size = int(samplenum/2.)),
                     [0]*(int(samplenum/2.)))
np.random.shuffle(mig_rate)

# select window width 
win_width = [.05]*samplenum # if varying: np.random.uniform(0.03,0.1,size = 1000000)
# select window center 
win_center = [.5]*samplenum # if varying: np.random.uniform(0.1, 0.9, size = 1000000)
# get number of replicates
#num_reps_list = [10000]*samplenum # if varying: (5000+np.random.randint(1,15000,1000000))*2

sample = 0 # usually this is written as a loop, where the sample variable is the loop number
length = 10000
Ne = 1000000
mutation_rate = 7e-8
num_replicates = 1
recombination_rate = 1e-8

# Four tips of the tree
population_configurations = [msprime.PopulationConfiguration(sample_size=1,initial_size=250000),
                            msprime.PopulationConfiguration(sample_size=1,initial_size=250000),
                            msprime.PopulationConfiguration(sample_size=1,initial_size=250000),
                            msprime.PopulationConfiguration(sample_size=1,initial_size=250000),]

# No migration initially
migration_matrix = [[0.,0.,0.,0.],
                    [0.,0.,0.,0.],
                    [0.,0.,0.,0.],
                    [0.,0.,0.,0.],]

# Define demographic events for msprime
demographic_events = [msprime.MigrationRateChange(time=0,
                                                  rate=0),
                     msprime.MigrationRateChange(time=(win_center[sample]-win_width[sample]/2.)*4*Ne,
                                                 rate=mig_rate[sample],
                                                 matrix_index=[0,2]),
                     msprime.MigrationRateChange(time=(win_center[sample]+win_width[sample]/2.)*4*Ne,
                                                 rate=0),
                     msprime.MassMigration(destination=1,
                                           source=0,
                                           time=1.0*4*Ne,
                                           proportion=1),
                     msprime.MassMigration(destination=1, 
                                           source=2,
                                           time=1.2*4*Ne,
                                           proportion=1),
                     msprime.MassMigration(destination=1, 
                                           source=3,
                                           time=1.5*4*Ne,
                                           proportion=1)
                     ]

# Our msprime simulation:
simulation = msprime.simulate(length=length,
                 Ne=Ne,
                 mutation_rate=mutation_rate,
                 num_replicates=num_replicates,
                 recombination_rate=recombination_rate,
                 population_configurations=population_configurations,
                 migration_matrix = migration_matrix,         
                 demographic_events=demographic_events)

fullseq = None
all_labels = np.empty(shape = [0,1])
all_seqs = list()
for currsim in simulation:
    simtrees = currsim.trees()
    for currtree in simtrees:
        phylo = dendropy.TreeList.get(data = currtree.newick(),schema='newick')
        s = seqgen.SeqGen()
        s.seq_len = round(currtree.get_interval()[1]-currtree.get_interval()[0])
        s.scale_branch_lens = 0.0000001 # will have to consider how best to use this
        dnamat=s.generate(phylo)
        charmat=dnamat.char_matrices[0]
        for lab,seq in charmat.items():
            all_labels = np.append(all_labels,lab.label)
            all_seqs.append(seq.symbols_as_list())

len(np.concatenate(np.array(all_seqs)[all_labels=='4']))

currsim.get_num_trees()

seqs = np.vstack([np.concatenate(np.array(all_seqs)[all_labels == '1']),
                  np.concatenate(np.array(all_seqs)[all_labels == '2']),
                  np.concatenate(np.array(all_seqs)[all_labels == '3']),
                  np.concatenate(np.array(all_seqs)[all_labels == '4'])])
seqs = np.array([[ord(i) for i in q] for q in seqs])

# A=65, T=84, G=71, C=67 -> A=0, T=1, G=2, C=3
seqs[seqs == 65] = 0
seqs[seqs == 84] = 1
seqs[seqs == 71] = 2
seqs[seqs == 67] = 3

#AA, AT, AG, AC,  -> AA, AT, AG, AC, TA, TT, TG, TC, GA, GT, GG, GC, CA, CT, CG, CC
#TA, TT, TG, TC, 
#GA, GT, GG, GC, 
#CA, CT, CG, CC

index_1616 = np.array(range(16)).reshape([4,4])
matrix1616 = np.zeros(shape = [16,16]).astype(int)
for column in range(len(seqs[0])):
    current4=seqs[:,column]
    row1616 = int(index_1616[current4[0],current4[1]])
    column1616 = int(index_1616[current4[2],current4[3]])
    matrix1616[row1616,column1616] = int(matrix1616[row1616,column1616] + int(1))

ml_feats = np.vstack([ml_feats, matrix1616.reshape([1,256])[0]])

ml_feats

ml_feats = np.empty(shape = [0,256]).astype(int) # holds our data
ml_labels = np.empty(shape = [0,1]) # holds our labels

samplenum = 10000 # how many simulation samples
# make list of migration rates
mig_rate = np.append(np.random.gamma(shape = 1,  # make half of the rates positive and the other half zero
                                     scale=.00001,
                                     size = int(samplenum/2.)),
                     [0]*(int(samplenum/2.)))
np.random.shuffle(mig_rate)

# select window width 
win_width = [.05]*samplenum # if varying: np.random.uniform(0.03,0.1,size = 1000000)
# select window center 
win_center = [.5]*samplenum # if varying: np.random.uniform(0.1, 0.9, size = 1000000)
# get number of replicates
#num_reps_list = [10000]*samplenum # if varying: (5000+np.random.randint(1,15000,1000000))*2

for currsample in range(samplenum):
    sample = currsample # usually this is written as a loop, where the sample variable is the loop number
    length = 10000
    Ne = 1000000
    mutation_rate = 7e-8
    num_replicates = 1
    recombination_rate = 1e-8

    # Four tips of the tree
    population_configurations = [msprime.PopulationConfiguration(sample_size=1,initial_size=250000),
                                msprime.PopulationConfiguration(sample_size=1,initial_size=250000),
                                msprime.PopulationConfiguration(sample_size=1,initial_size=250000),
                                msprime.PopulationConfiguration(sample_size=1,initial_size=250000),]

    # No migration initially
    migration_matrix = [[0.,0.,0.,0.],
                        [0.,0.,0.,0.],
                        [0.,0.,0.,0.],
                        [0.,0.,0.,0.],]

    # Define demographic events for msprime
    demographic_events = [msprime.MigrationRateChange(time=0,
                                                      rate=0),
                         msprime.MigrationRateChange(time=(win_center[sample]-win_width[sample]/2.)*4*Ne,
                                                     rate=mig_rate[sample],
                                                     matrix_index=[0,2]),
                         msprime.MigrationRateChange(time=(win_center[sample]+win_width[sample]/2.)*4*Ne,
                                                     rate=0),
                         msprime.MassMigration(destination=1,
                                               source=0,
                                               time=1.0*4*Ne,
                                               proportion=1),
                         msprime.MassMigration(destination=1, 
                                               source=2,
                                               time=1.2*4*Ne,
                                               proportion=1),
                         msprime.MassMigration(destination=1, 
                                               source=3,
                                               time=1.5*4*Ne,
                                               proportion=1)
                         ]

    # Our msprime simulation:
    simulation = msprime.simulate(length=length,
                     Ne=Ne,
                     mutation_rate=mutation_rate,
                     num_replicates=num_replicates,
                     recombination_rate=recombination_rate,
                     population_configurations=population_configurations,
                     migration_matrix = migration_matrix,         
                     demographic_events=demographic_events)
    fullseq = None
    seqlength = 15
    all_labels = np.empty(shape = [0,1])
    all_seqs = list()
    for currsim in simulation:
        simtrees = currsim.trees()
        for currtree in simtrees:
            phylo = dendropy.TreeList.get(data = currtree.newick(),schema='newick')
            s = seqgen.SeqGen()
            s.seq_len = int(round(currtree.get_interval()[1]-currtree.get_interval()[0]))
            s.scale_branch_lens = 0.0000001 # will have to consider how best to use this
            dnamat=s.generate(phylo)
            charmat=dnamat.char_matrices[0]
            for lab,seq in charmat.items():
                all_labels = np.append(all_labels,lab.label)
                all_seqs.append(seq.symbols_as_list())
    seqs = np.vstack([np.concatenate(np.array(all_seqs)[all_labels == '1']),
                      np.concatenate(np.array(all_seqs)[all_labels == '2']),
                      np.concatenate(np.array(all_seqs)[all_labels == '3']),
                      np.concatenate(np.array(all_seqs)[all_labels == '4'])])
    seqs = np.array([[ord(i) for i in q] for q in seqs])
    # A=65, T=84, G=71, C=67 -> A=0, T=1, G=2, C=3
    seqs[seqs == 65] = 0
    seqs[seqs == 84] = 1
    seqs[seqs == 71] = 2
    seqs[seqs == 67] = 3

    index_1616 = np.array(range(16)).reshape([4,4])
    matrix1616 = np.zeros(shape = [16,16]).astype(int)
    for column in range(len(seqs[0])):
        current4=seqs[:,column]
        row1616 = int(index_1616[current4[0],current4[1]])
        column1616 = int(index_1616[current4[2],current4[3]])
        matrix1616[row1616,column1616] = int(matrix1616[row1616,column1616] + int(1))

    ml_feats = np.vstack([ml_feats, matrix1616.reshape([1,256])[0]])
    ml_labels = np.vstack([ml_labels, mig_rate[sample]])
    print(sample)
    print(mig_rate[sample])

import h5py
f = h5py.File("sim_counts2.hdf5", "w")
f.create_dataset("labels", (len(ml_labels),))
f.create_dataset("matrices",(ml_feats.shape),dtype = 'i')
f['labels'][...] = ml_labels.reshape(1,-1)[0]
f['matrices'][...] = ml_feats
f.close()

norm=preprocessing.Normalizer(copy=False,norm='l1')
trans_norm = norm.transform(ml_feats)

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(ml_feats)  
scaled = scaler.transform(ml_feats)

int_mig_rate=(mig_rate > 0).astype(int)

from sklearn import svm
X = scaled[0:1000]
y = int_mig_rate[0:1000]
clf = svm.SVC(C=10000)
clf.fit(X, y)  

testset = np.empty(shape=[0,257])
for i in range(1000,1100):
    testset=np.vstack([testset,np.append(mig_rate[i],scaled[i])])

testset[:,0] = (testset[:,0] > 0).astype(int)
sorted_test= testset[testset[:,0].argsort()]

plt.plot(sorted_test[:,0], 'ro')
plt.show()

plt.plot(clf.predict(sorted_test[:,1:257]),'ro')
plt.show()

from sklearn import svm
X = trans_norm[0:1000]
y = mig_rate[0:1000]
regr = svm.SVR(C=1000,epsilon=.01)
regr.fit(X, y)  

testset = np.empty(shape=[0,257])
for i in range(1000,1100):
    testset=np.vstack([testset,np.append(mig_rate[i],trans_norm[i])])

#testset[:,0] = (testset[:,0] > 0).astype(int)
sorted_test= testset[testset[:,0].argsort()]

plt.plot(sorted_test[:,0], 'ro')
plt.show()

plt.plot(regr.predict(sorted_test[:,1:257]),'ro')
plt.show()

from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components=2, kernel = 'rbf', gamma = .0004)

X_red = rbf_pca.fit_transform(trans_norm)

zeros = X_red[(int_mig_rate==0)[0:len(trans_norm)],:]
ones = X_red[(int_mig_rate==1)[0:len(trans_norm)],:]

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(zeros[:,0]*10,zeros[:,1]*10, s=10, c='b', marker="s", label='no migration')
ax1.scatter(ones[:,0]*10,ones[:,1]*10, s=10, c='r', marker="o", label='with migration')
plt.legend(loc='upper left');
plt.title("PCA: migration vs. no migration")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(trans_norm,ml_labels.reshape(1,-1)[0])

gbrt=GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)
bst_n_estimators

gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=12)
gbrt_best.fit(trans_norm[0:1000], ml_labels.reshape(1,-1)[0][0:1000])

plt.plot(sorted_test[:,0], 'ro')
plt.show()

plt.plot(gbrt_best.predict(sorted_test[:,1:257]),'ro')
plt.show()

from sklearn.neural_network import MLPRegressor

neuralnet = MLPRegressor(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(12,3), random_state=7, max_iter=100000,
                        tol=1e-1,momentum = 0.2,batch_size = 100, power_t = .5)
X = trans_norm[0:1000]
y = mig_rate[0:1000]
neuralnet.fit(X,y)

plt.plot(neuralnet.predict(sorted_test[:,1:257]),'ro')
plt.show()



