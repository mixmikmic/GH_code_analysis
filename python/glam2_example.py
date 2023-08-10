get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('matplotlib inline')
from glam2_wrapper import Glam2

from utilities import Weblogo
wl = Weblogo(color_scheme='classic')

glam2 = Glam2(alphabet='dna',
              gap_in_alphabet=True,
              scoring_criteria='pwm',
              alignment_runs=5,
             
              weblogo_obj = wl)

glam2.fit(fasta_file="seq18.fa")

for i in glam2.original_motives_list:
    for j in i:
        print j
    print

for i in glam2.aligned_motives_list:
    for j in i:
        print j
    print 

for i in glam2.motives_list:
    for j in i:
        print j
    print

predictions = glam2.predict(input_seqs='seq18.fa', return_list=True)
for p in predictions: print p

predictions = glam2.predict(input_seqs="seq9.fa", return_list=False)
for p in predictions: print p

match = glam2.transform(input_seqs='seq9.fa', return_match=True)
for m in match: print m

match = glam2.transform(input_seqs='seq9.fa', return_match=False)
for m in match: print m

glam_2 = Glam2(alphabet='dna', gap_in_alphabet=True, scoring_criteria='pwm', alignment_runs=6)

predictions = glam_2.fit_predict(fasta_file='seq9.fa', return_list=True)
for p in predictions: print p

matches = glam_2.fit_transform(fasta_file='seq9.fa', return_match=True)
for m in matches: print m

#printing motives as lists
for motif in glam2.motives_list:
    for m in motif:
        print m
    print

glam2.display_logo(do_alignment=False)

glam2.display_logo(motif_num=1)

glam2.align_motives()    #MSA with Muscle
motives1=glam2.aligned_motives_list
for m in motives1:
    for i in m:
        print i
    print

glam2.display_logo(do_alignment=True)

glam2.display()

glam2.matrix()

glam2.display(motif_num=3)

test_seq = 'GGAGAAAATACCGC' * 10
seq_score = glam2.score(motif_num=2, seq=test_seq)
print seq_score

glam_3 = Glam2(alphabet='dna', gap_in_alphabet=True, scoring_criteria='hmm', alignment_runs=3)
matches = glam_3.fit_transform(fasta_file="seq9.fa", return_match=True)
for m in matches: print m



