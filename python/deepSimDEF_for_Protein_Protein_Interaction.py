import os
import sys
import random
import operator
import numpy as np
import keras.backend as K

from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

from deepSimDEF.tools.PPI_data_provider import gene_pair_data_reader, input_data_maker
from deepSimDEF.tools.PPI_model_saver import save_model, save_embeddings
from deepSimDEF.netwroks.PPI_network import PPI_model_builder

np.random.seed(321)

FOLD = 10
DROPOUT = 0.3
MAX_POOL = True

PRE_TRAINED = True
UPDATABLE = True

ACTIVATION_HIDDEN = 'relu'
ACTIVATION_HIGHWAY = 'sigmoid'
ACTIVATION_OUTPUT = 'sigmoid'

EMBEDDING_DIM = 100
NB_EPOCH = 20
BATCH_SIZE = 256
OPTIMIZER = 'adadelta'

IEA = True
SEQ = False

TRANSFER_LEARNING = False

SAVE_MODEL = True
SAVE_EMBEDDINGS = True

SUB_ONTOLOGY = ['BP', 'CC', 'MF']
SUB_ONTOLOGY_work = ['BP', 'CC', 'MF']

WITH_HIGH_THROUPUT = False

SBOs = {}
for sbo in SUB_ONTOLOGY_work:
    if sbo == 'BP':
        SBOs[sbo] = 'Biolobical Process (BP)'
    elif sbo == 'CC':
        SBOs[sbo] = 'Cellular Component (CC)'
    elif sbo == 'MF':
        SBOs[sbo] = 'Molecular Function (MF)'
    
WE = {}
embedding_save = {}
MAX_SEQUENCE_LENGTH = {}
MAX_SEQUENCE_LENGTH_INDEX = {}
sequences = {}
word_indeces = {}
protein_index = {}    
    
for sbo in SUB_ONTOLOGY:
    WE[sbo] = 'deepSimDEF/embeddings/GO_' + sbo + '_Embeddings_100D.emb'
    embedding_save[sbo] = 'GO_' + sbo + '_Embeddings_100D_Updated'
    MAX_SEQUENCE_LENGTH[sbo] = 0
    MAX_SEQUENCE_LENGTH_INDEX[sbo] = []
    sequences[sbo] = []
    word_indeces[sbo] = []
    protein_index[sbo] = {}
    
    if IEA:
        file_reader = open('deepSimDEF/gene_annotations/gene_product_GO_terms_with_IEA' + '.' + sbo)
    else:
        file_reader = open('deepSimDEF/gene_annotations/gene_product_GO_terms_without_IEA' + '.' + sbo)
    
    index_counter = 1
    texts = []
    for line in file_reader:
        values = line.rstrip().replace(':', '').split()
        protein_index[sbo][values[0]] = index_counter
        if len(values[1:]) > MAX_SEQUENCE_LENGTH[sbo]:
            MAX_SEQUENCE_LENGTH[sbo] = len(values[1:])
            MAX_SEQUENCE_LENGTH_INDEX[sbo] = index_counter
        texts.append(' '.join(values[1:]))
        index_counter += 1
        
    tokenizer = Tokenizer(lower=False, num_words=0)
    tokenizer.fit_on_texts(texts)
    sequences[sbo] = tokenizer.texts_to_sequences(texts)

    word_indeces[sbo] = tokenizer.word_index
    
    if sbo == 'BP':
        print "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Biolobical Process (BP) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    elif sbo == 'CC':
        print "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cellular Component (CC) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    elif sbo == 'MF':
        print "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Molecular Function (MF) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            
    print "Found " + str(len(word_indeces[sbo])) + " unique tokens in " + sbo

    MOST_FREQUENT_LEVEL = 10
    print 'Top', MOST_FREQUENT_LEVEL, 'Most Frequent GO terms annotating sequences in', sbo + ":"
    for GO_ID, indx in sorted(word_indeces[sbo].items(), key=operator.itemgetter(1))[:MOST_FREQUENT_LEVEL]:
        print '  >>>', GO_ID, '   ' ,indx
        
    print "Number of annotated gene products by '" + sbo + "' terms: " + str(len(sequences[sbo]))
    print "Maximum annotation length of one gene product ('" + sbo + "' sub-ontology):", MAX_SEQUENCE_LENGTH[sbo]
    print "Index/line of the gene product with maximum annotations ('" + sbo + "' sub-ontology):", MAX_SEQUENCE_LENGTH_INDEX[sbo]
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    
    file_reader.close()
    
    
fully_annotated_sequences = []   # we keep only those genes for which we have annatoation from all ontologies (defined in SUB_ONTOLOGY variable)
for sbo in SUB_ONTOLOGY:
    fully_annotated_sequences.append(protein_index[sbo].keys())
fully_annotated_sequences = list(set(fully_annotated_sequences[0]).intersection(*fully_annotated_sequences))
print "Number of fully annotated gene products:", len(fully_annotated_sequences)


input_data_dir = 'deepSimDEF/datasets/PPI_data/PPI_FULL_physical_interactions_manually_curated'
annotation_G1_dic_MC, annotation_G2_dic_MC, interaction_pr_list_MC = gene_pair_data_reader(data_dir=input_data_dir, 
                                                                                           SUB_ONTOLOGY_work=SUB_ONTOLOGY_work, 
                                                                                           fully_annotated_sequences=fully_annotated_sequences, 
                                                                                           sequences=sequences, 
                                                                                           protein_index=protein_index,
                                                                                           MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)


VALIDATION_SPLIT = 1.0/FOLD
indices = np.arange(annotation_G1_dic_MC[sbo].shape[0])
np.random.shuffle(indices)
test_size = int(VALIDATION_SPLIT * annotation_G1_dic_MC[sbo].shape[0])


annotation_G1_dic_HT = []
annotation_G2_dic_HT = []
interaction_pr_list_HT = []

if WITH_HIGH_THROUPUT:
    input_data_dir = 'deepSimDEF/datasets/PPI_data/PPI_FULL_physical_interactions_high_throughput'
    annotation_G1_dic_HT, annotation_G2_dic_HT, interaction_pr_list_HT = gene_pair_data_reader(data_dir=input_data_dir, 
                                                                                               SUB_ONTOLOGY_work=SUB_ONTOLOGY_work, 
                                                                                               fully_annotated_sequences=fully_annotated_sequences, 
                                                                                               sequences=sequences, 
                                                                                               protein_index=protein_index,
                                                                                               MAX_SEQUENCE_LENGTH= MAX_SEQUENCE_LENGTH)

for sbo in SUB_ONTOLOGY_work:
    print "@@@ " + SBOs[sbo] + " @@@"

if IEA:
    print "^^^ With IEA ^^^"
else:
    print "^^^ Without IEA ^^^"

print "%%% Optimizer:", OPTIMIZER, "%%%"

if PRE_TRAINED:
    if UPDATABLE:
        print "+++ Pre-trained (updatable) +++"
    else:
        print "+++ Pre-trained (not updatable) +++"
else:
    print "+++ NOT Pre-trained +++"

models = []
embedding_layers = []
bests = []
thresholds = []
B = []

for m in range(0, FOLD):
    network = PPI_model_builder(EMBEDDING_DIM, 
                                 model_ind=m, 
                                 MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, 
                                 WORD_EMBEDDINGS=WE,
                                 SUB_ONTOLOGY_work=SUB_ONTOLOGY_work,
                                 word_indeces=word_indeces, 
                                 ACTIVATION_HIDDEN=ACTIVATION_HIDDEN, 
                                 ACTIVATION_HIGHWAY=ACTIVATION_HIGHWAY, 
                                 ACTIVATION_OUTPUT=ACTIVATION_OUTPUT, 
                                 DROPOUT=DROPOUT, 
                                 OPTIMIZER=OPTIMIZER)
    models.append(network[0])
    embedding_layers.append(network[1])
    bests.append(0)
    thresholds.append(0)
    B.append({})

RES = {}
best_total_f1 = 0
best_threshold = 0

early_stopping = EarlyStopping(monitor='val_loss', patience = 3)
cor = {}

best_epoch = 0

def pred(A, treshold = 0.5):
    B = []
    for n in A:
        if treshold < n:
            B.append(1)
        else:
            B.append(0)
    return B

def run_my_model(model_index, seq):
    X_train, y_train, X_test, y_test = input_data_maker(model_id=model_index, 
                                                        test_size=test_size, 
                                                        indices=indices, 
                                                        annotation_G1_dic_MC=annotation_G1_dic_MC, 
                                                        annotation_G2_dic_MC=annotation_G2_dic_MC, 
                                                        interaction_pr_list_MC=interaction_pr_list_MC, 
                                                        annotation_G1_dic_HT=annotation_G1_dic_HT,
                                                        annotation_G2_dic_HT=annotation_G2_dic_HT,
                                                        interaction_pr_list_HT=interaction_pr_list_HT,
                                                        SUB_ONTOLOGY_work=SUB_ONTOLOGY_work,
                                                        WITH_HIGH_THROUPUT=False)
    model = models[model_index]
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=1, validation_data=(X_test,y_test))
    p =  model.predict(X_test)
    for i in seq:
        predictions = np.asarray(pred(p, i))
        B[model_index][i] = np.round(f1_score(y_test, predictions, average='binary'), 5)
    pr = max(B[model_index].iteritems(), key=operator.itemgetter(1))[1]
    thresholds[model_index] = max(B[model_index].iteritems(), key=operator.itemgetter(1))[0]
    st = ''
    b = bests[model_index]
    if bests[model_index] < pr: 
        bests[model_index] = pr
        treshold = thresholds[model_index]
        st = "+ " + str(bests[model_index])
    else:
        st = "- " + str(bests[model_index])
    print ">>> F1-score (" + str(model_index + 1) + "):", pr, "Best (" + str(model_index + 1) + "):", st, "(" + str(thresholds[model_index]) + " : " + str(np.round(pr - b, 5)) + ")" + "\n"

def get_results(epoch_no):
    for i in seq:
        RES[i] = 0
        for j in range(FOLD):
            RES[i] += B[j][i]/FOLD
    res = max(RES.iteritems(), key=operator.itemgetter(1))[1]
    threshold_res = max(RES.iteritems(), key=operator.itemgetter(1))[0]
    cor[epoch_no + 1] = res
    total_max = 0
    for i, j in sorted(cor.items(), key=operator.itemgetter(1)):
        if total_max < j:
            total_max = j
            best_epoch = i
            threshold_best = threshold_res
    
    print "F1-score for this epoch:", res, "(", threshold_res, ")-- Best F1-score::==>", str(total_max), "(", threshold_best, ")  (for epoch #", str(best_epoch), "of", str(NB_EPOCH), "epochs)" + "\n"

def get_final_result():
    final_max = 0
    best_epoch = 0
    for i, j in sorted(cor.items(), key=operator.itemgetter(1)):
        if final_max < j:
            final_max = j
            best_epoch = i
        
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FINAL RESULT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~" + "\n" 
    print "For embedding size '" + str(EMBEDDING_DIM) + "' best number of epochs is '" + str(i) + "' with F1-score of: " + str(final_max) +"\n"
    
for e in range(NB_EPOCH):
     
    print "~~~~~~~~~ " + '/'.join(SUB_ONTOLOGY_work) +" ~~~~~~~~~~~~~~ EPOCH " + str(e + 1) + "/" + str(NB_EPOCH) + " (Embedding dimention: " + str(EMBEDDING_DIM) + ") ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n" 
    seq = [0.5]
    if SEQ:
        seq = np.arange(0.11, 0.9, 0.01)
        
    for index in range(0, len(models)):
        run_my_model(index, seq)
    
    get_results(e)

get_final_result()

if SAVE_MODEL:
    save_model(FOLD=FOLD, models=models)
    
if SAVE_EMBEDDINGS:
    save_embeddings(FOLD=FOLD, 
                           embedding_layers=embedding_layers,
                           word_indeces=word_indeces, 
                           SUB_ONTOLOGY_work=SUB_ONTOLOGY_work,
                           embedding_save=embedding_save)
    



