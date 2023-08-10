get_ipython().magic('matplotlib inline')
from meme_wrapper import Meme
import logging

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

# saving data as fasta files
with open('seq18.fa','w') as f_train:
    for seq in train:
        f_train.write('>' + seq[0] + ' \n')
        f_train.write(seq[1] + '\n')
        
with open('seq9.fa','w') as f_test:
    for seq in test:
        f_test.write('>' + seq[0] + ' \n')
        f_test.write(seq[1] + '\n')

# Meme().display_meme_help()
from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=2)

from utilities import Weblogo
wl = Weblogo(color_scheme='classic')
meme1 = Meme(alphabet="dna",    # {ACGT}
             gap_in_alphabet=False,
             mod="anr",     # Any number of repititions
             output_dir="meme_anr", 
             nmotifs=3,    # Number of motives to be found
             
             weblogo_obj = wl
            )

meme1.fit(fasta_file="seq18.fa")

predictions = meme1.predict(input_seqs=test, return_list=True)
for p in predictions: print p

predictions = meme1.predict(input_seqs="seq9.fa", return_list=False)
for p in predictions: print p

match = meme1.transform(input_seqs=test, return_match=True)
for m in match: print m

match = meme1.transform(input_seqs=test, return_match=False)
for m in match: print m

print meme1.e_values

meme2 = Meme(alphabet="dna", mod="anr", nmotifs=3)

predictions = meme2.fit_predict(fasta_file="seq18.fa", return_list=True)
for p in predictions: print p

matches = meme2.fit_transform(fasta_file="seq18.fa", return_match=True)
for m in matches: print m

#printing motives as lists
for motif in meme1.motives_list:
    for m in motif:
        print m
    print

meme1.display_logo(do_alignment=False)

meme1.display_logo(motif_num=1)

meme1.align_motives()    #MSA with Muscle
motives1=meme1.aligned_motives_list
for m in motives1:
    for i in m:
        print i
    print

meme1.display_logo(do_alignment=True)

meme1.display()

meme1.matrix()

meme1.display(motif_num=3)

test_seq = 'GGAGAAAATACCGC' * 10
seq_score = meme1.score(motif_num=2, seq=test_seq)
print seq_score

meme2 = Meme(alphabet="dna", scoring_criteria="hmm", k=1, threshold=1.0,mod="anr", nmotifs=3, minw=7, maxw=9)
matches = meme2.fit_transform(fasta_file="seq9.fa", return_match=True)
for m in matches: print m

get_ipython().run_cell_magic('time', '', '# Markov Model score\nmm_score = meme2.score(motif_num=2, seq="ACGT"*10)\nprint mm_score')



