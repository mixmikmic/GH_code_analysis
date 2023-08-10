get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from dreme_wrapper import Dreme

dreme = Dreme(alphabet='dna',
              e_threshold=1e-9,
              scoring_criteria='pwm',
              mink=4,
              maxk=8,
              output_dir='dreme_output')

get_ipython().run_cell_magic('time', '', "dreme.fit(fasta_file='dreme-sample.fa')")

print "Version:", dreme.record.version

print "sequence alphabet: ", dreme.record.alphabet

print "Consensus Sequences"
for i, c in enumerate(dreme.record.consensus_seqs):
    print "Motif %d: "%(i+1), c
    print "Width:", dreme.record.widths[i],
    print "Nsites: ", dreme.record.nsites[i],
    print "E-Value: ", dreme.record.e_values[i]
    print 


for i, mat in enumerate(dreme.record.prob_matrices):
    for m in mat:
        print m
    print

dreme.motives_list

for k, i in enumerate(dreme.motives_list):
    for l, j in enumerate(i):
        if '-' in j[1]:
            print k, l, j

predictions = dreme.predict(input_seqs="dreme-sample_2.fa", return_list=True)
for p in predictions: print p

predictions = dreme.predict(input_seqs="dreme-sample_2.fa", return_list=False)
for p in predictions: print p

match = dreme.transform(input_seqs="dreme-sample_2.fa", return_match=True)
for m in match: print m

match = dreme.transform(input_seqs="dreme-sample_2.fa", return_match=False)
for m in match: print m

dreme.display_logo(do_alignment=False)



