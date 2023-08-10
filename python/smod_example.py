get_ipython().magic('matplotlib inline')
from smod_wrapper import SMoDWrapper
from utilities import Weblogo

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from eden.util import configure_logging
import logging
logger = logging.getLogger()
configure_logging(logger,verbosity=2)



train = [
    ('ce1cg', 
     'TAATGTTTGTGCTGGTTTTTGTGGCATCGGGCGAGAATAGCGCGTGGTGTGAAAGACTGTTTTTTTGATCGTTTTCACAAAAATGGAAGTCCACAGTCTTGACAG'),
    ('ara', 
     'GACAAAAACGCGTAACAAAAGTGTCTATAATCACGGCAGAAAAGTCCACATTGATTATTTGCACGGCGTCACACTTTGCTATGCCATAGCATTTTTATCCATAAG'),
    ('bglr1', 
     'ACAAATCCCAATAACTTAATTATTGGGATTTGTTATATATAACTTTATAAATTCCTAAAATTACACAAAGTTAATAACTGTGAGCATGGTCATATTTTTATCAAT'),
    ('crp', 
     'CACAAAGCGAAAGCTATGCTAAAACAGTCAGGATGCTACAGTAATACATTGATGTACTGCATGTATGCAAAGGACGTCACATTACCGTGCAGTACAGTTGATAGC'),
    ('cya', 
     'ACGGTGCTACACTTGTATGTAGCGCATCTTTCTTTACGGTCAATCAGCAAGGTGTTAAATTGATCACGTTTTAGACCATTTTTTCGTCGTGAAACTAAAAAAACC'),
    ('deop2', 
     'AGTGAATTATTTGAACCAGATCGCATTACAGTGATGCAAACTTGTAAGTAGATTTCCTTAATTGTGATGTGTATCGAAGTGTGTTGCGGAGTAGATGTTAGAATA'),
    ('gale', 
     'GCGCATAAAAAACGGCTAAATTCTTGTGTAAACGATTCCACTAATTTATTCCATGTCACACTTTTCGCATCTTTGTTATGCTATGGTTATTTCATACCATAAGCC'),
    ('ilv', 
     'GCTCCGGCGGGGTTTTTTGTTATCTGCAATTCAGTACAAAACGTGATCAACCCCTCAATTTTCCCTTTGCTGAAAAATTTTCCATTGTCTCCCCTGTAAAGCTGT'),
    ('lac', 
     'AACGCAATTAATGTGAGTTAGCTCACTCATTAGGCACCCCAGGCTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGGAATTGTGAGCGGATAACAATTTCAC'),
    ('male', 
     'ACATTACCGCCAATTCTGTAACAGAGATCACACAAAGCGACGGTGGGGCGTAGGGGCAAGGAGGATGGAAAGAGGTTGCCGTATAAAGAAACTAGAGTCCGTTTA'),
    ('malk', 
     'GGAGGAGGCGGGAGGATGAGAACACGGCTTCTGTGAACTAAACCGAGGTCATGTAAGGAATTTCGTGATGTTGCTTGCAAAAATCGTGGCGATTTTATGTGCGCA'),
    ('malt', 
     'GATCAGCGTCGTTTTAGGTGAGTTGTTAATAAAGATTTGGAATTGTGACACAGTGCAAATTCAGACACATAAAAAAACGTCATCGCTTGCATTAGAAAGGTTTCT'),
    ('ompa', 
     'GCTGACAAAAAAGATTAAACATACCTTATACAAGACTTTTTTTTCATATGCCTGACGGAGTTCACACTTGTAAGTTTTCAACTACGTTGTAGACTTTACATCGCC'),
    ('tnaa', 
     'TTTTTTAAACATTAAAATTCTTACGTAATTTATAATCTTTAAAAAAAGCATTTAATATTGCTCCCCGAACGATTGTGATTCGATTCACATTTAAACAATTTCAGA'),
    ('uxu1', 
     'CCCATGAGAGTGAAATTGTTGTGATGTGGTTAACCCAATTAGAATTCGGGATTGACATGTCTTACCAAAAGGTAGAACTTATACGCCATCTCATCCGATGCAAGC'),
    ('pbr322', 
     'CTGGCTTAACTATGCGGCATCAGAGCAGATTGTACTGAGAGTGCACCATATGCGGTGTGAAATACCGCACAGATGCGTAAGGAGAAAATACCGCATCAGGCGCTC'),
    ('trn9cat', 
     'CTGTGACGGAAGATCACTTCGCAGAATAAATAAATCCTGGTGTCCCTGTTGATACCGGGAAGCCCTGGGCCAACTTTTGGCGAAAATGAGACGTTGATCGGCACG'),
    ('tdc', 
     'GATTTTTATACTTTAACTTGTTGATATTTAAAGGTATTTAATTGTAATAACGATACTCTGGAAAGTATTGAAAGTTAATTTGTGAGTGGTCGCACATATCCTGTT'),
    ]

# test data consists of first 9 sequences of training data
test = train[:9]

from sklearn.cluster import KMeans
km = KMeans()

wl=Weblogo(color_scheme = 'classic')

smod = SMoDWrapper(alphabet='dna',
                 complexity = 3,
                 n_clusters = 3 * 3,
                 clusterer = KMeans(),
                 pos_block_size = 5,
                 neg_block_size = 5,
                 min_score=4,
                 min_freq=0.5,
                 min_cluster_size=10,
                 similarity_th=.5,
                 freq_th=0.03,
                 weblogo_obj=wl)

smod.fit(seqs=train)

for i in smod.original_motives_list:
    for j in i:
        print j
    print

for i in smod.aligned_motives_list:
    for j in i:
        print j
    print 

for i in smod.motives_list:
    for j in i:
        print j
    print

predictions = smod.predict(input_seqs=test, return_list=True)
for p in predictions: print p

predictions = smod.predict(input_seqs=test, return_list=False)
for p in predictions: print p

match = smod.transform(input_seqs=test, return_match=True)
for m in match: print m

match = smod.transform(input_seqs=test, return_match=False)
for m in match: print m

smod2= SMoDWrapper(alphabet='dna',
                 complexity = 3,
                 n_clusters = 3 * 3,
                 clusterer = KMeans(),
                 pos_block_size = 5,
                 neg_block_size = 5,
                 weblogo_obj=wl)
predictions = smod2.fit_predict(seqs=train)
for p in predictions: print p

smod2= SMoDWrapper(alphabet='dna',
                 complexity = 3,
                 n_clusters = 3 * 3,
                 clusterer = KMeans(),
                 pos_block_size = 5,
                 neg_block_size = 5,
                 weblogo_obj=wl)
matches = smod2.fit_transform(seqs=train)
for m in matches: print m

#printing motives as lists
for motif in smod.motives_list:
    for m in motif:
        print m
    print

smod.display_logo(do_alignment=False)

smod.display_logo(motif_num=1)

smod.align_motives()    #MSA with Muscle
motives1=smod.aligned_motives_list
for m in motives1:
    for i in m:
        print i
    print

smod.display()

smod.display(motif_num=2)

# Score a test sequence using probability score
test_seq = 'ACGT' * 10
seq_score = smod.score(motif_num=1, seq=test_seq)
print seq_score

meme2 = SMoDWrapper(alphabet="dna",
                    scoring_criteria="hmm", 
                    k=1, 
                    threshold=0.5,
                    complexity = 3,
                    n_clusters = 3 * 3,
                    clusterer = KMeans(),
                    pos_block_size = 5,
                    neg_block_size = 5,
                    min_score=4,
                    min_freq=0.5,
                    min_cluster_size=10,
                    similarity_th=.5,
                    freq_th=0.03)
matches = meme2.fit_transform(seqs=train, return_match=True)
for m in matches: print m

get_ipython().run_cell_magic('time', '', '# Markov Model score\nmm_score = smod.score(motif_num=2, seq="ACGT"*10)\nprint mm_score')



