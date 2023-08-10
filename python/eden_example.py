get_ipython().magic('matplotlib inline')
from eden_wrapper import EdenWrapper
from utilities import Weblogo

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
km = KMeans(n_clusters=3)

wl=Weblogo(color_scheme = 'classic')

ew = EdenWrapper(alphabet='dna',
                # distance=10, 
                # radius=5, 
                clustering_algorithm=km,
                threshold=0,
                
                weblogo_obj=wl)

ew.fit(seqs=train)

for i in ew.original_motives_list:
    for j in i:
        print j
    print

for i in ew.aligned_motives_list:
    for j in i:
        print j
    print 

for i in ew.motives_list:
    for j in i:
        print j
    print

predictions = ew.predict(input_seqs=test, return_list=True)
for p in predictions: print p

predictions = ew.predict(input_seqs=test, return_list=False)
for p in predictions: print p

match = ew.transform(input_seqs=test, return_match=True)
for m in match: print m

match = ew.transform(input_seqs=test, return_match=False)
for m in match: print m

ew2= EdenWrapper(alphabet='dna',
                   # distance=10, 
                   # radius=5, 
                   clustering_algorithm=km)
predictions = ew2.fit_predict(seqs=train)
for p in predictions: print p

matches = ew2.fit_transform(seqs=train)
for m in matches: print m

#printing motives as lists
for motif in ew.motives_list:
    for m in motif:
        print m
    print

ew.display_logo(do_alignment=False)

ew.display_logo(motif_num=1)

ew.align_motives()    #MSA with Muscle
motives1=ew.aligned_motives_list
for m in motives1:
    for i in m:
        print i
    print

ew.display()

ew.display(motif_num=3)

# Score a test sequence using probability score
test_seq = 'AAAAAAAAAAAA' * 10
seq_score = ew.score(motif_num=2, seq=test_seq)
print seq_score









get_ipython().run_cell_magic('time', '', "# Score a test sequence using Hidden Markov Model score\nmm_score = ew.score_mm(motif_num=1, seq=test_seq)\nprint 'Motif instances used for scoring:'\nfor m in ew.original_motives_list[0]:\n    print m[1]\nprint\nprint 'Score:'\nprint mm_score")



