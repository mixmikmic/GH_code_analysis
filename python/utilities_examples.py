from utilities import MotifWrapper, MuscleAlignWrapper, Weblogo
from IPython.display import Image, display
import numpy as np

# motives as list of lists - sample output from meme_wrapper

motives = [
    #Motif 1
    [('male', 'TGTAACAGAGATCACACAA'),
     ('ompa', 'CCTGACGGAGTTCACACTT'),
     ('lac', 'TGTGAGTTAGCTCACTCAT'),
     ('tdc', 'TGTGAGTGGTCGCACATAT'),
     ('pbr322', 'TGTGAAATACCGCACAGAT'),
     ('tnaa', 'TGTGATTCGATTCACATTT'),
     ('deop2', 'TTTGAACCAGATCGCATTA'),
     ('ce1cg', 'TTTGATCGTTTTCACAAAA'),
     ('ara', 'TTTGCACGGCGTCACACTT'),
     ('bglr1', 'TGTGAGCATGGTCATATTT'),
     ('crp', 'TGCAAAGGACGTCACATTA'),
     ('malt', 'TGTGACACAGTGCAAATTC'),
     ('gale', 'TGTAAACGATTCCACTAAT'),
     ('cya', 'TGTTAAATTGATCACGTTT'),
     ('uxu1', 'TGTGATGTGGTTAACCCAA'),
     ('ilv', 'CGTGATCAACCCCTCAATT'),
     ('gale', 'TGTCACACTTTTCGCATCT'),
     ('malk', 'CGTGATGTTGCTTGCAAAA')],
    
    #Motif 2
    [('pbr322', 'GGAGAAAATACCGC'),
     ('ce1cg', 'GGCGAGAATAGCGC'),
     ('gale', 'GCATAAAAAACGGC'),
     ('malk', 'GATGAGAACACGGC'),
     ('ara', 'GCAGAAAAGTCCAC'),
     ('trn9cat', 'GGCGAAAATGAGAC')],
    
    #Motif3
    [('lac', 'CCCCAGGCTTTACA'), 
     ('ce1cg', 'CCACAGTCTTGACA')]]

logos=[]
wb = Weblogo(output_format = 'png',    #[eps, png, png_print, jpeg]
             stacks_per_line = 40,
             sequence_type = 'dna',    #[protein, dna, rna]
             ignore_lower_case = False,
             units = 'bits',    #['bits', 'nats', 'digits', 'kT', 'kJ/mol', 'kcal/mol', 'probability']
             first_position = 1,
             #logo_range = list(),
             scale_stack_widths = True,
             error_bars = True,
             title = '',
             figure_label = '',
             show_x_axis = True,
             x_label = '',
             show_y_axis = True,
             y_label = '',
             y_axis_tic_spacing = 1.0,
             show_ends = False,
             color_scheme = 'classic', #[auto, base, pairing, charge, chemistry, classic, monochrome]
             resolution = 200,
             fineprint = ' ',
            )

for i in range(len(motives)):
    logo_image = wb.create_logo(seqs=motives[i])
    logos.append(logo_image)

for i in range(len(motives)):
    display(Image(logos[i]))

aligned_motives=[]
ma = MuscleAlignWrapper(diags=False, 
                         maxiters = 16, 
                         maxhours = None,
                        )

for i in range(len(motives)):
    aligned_motives.append( ma.transform(seqs=motives[i]) )

for i in range(len(motives)):
    print 'Motif %d:'%(i+1)
    for am in aligned_motives[i]: print am
    print

pwm1 = MotifWrapper(alphabet='dna', 
                    pseudocounts={'-':0, 'A':1, 'C':1, 'G':1, 'T':1})
pwm1.fit(motives)

pwm1.display()

pwm1.matrix()

test_seq = 'GGAGAAAATACCGC' * 10    # 140 characters
seq_score = pwm1.score_pwm(motif_num=2, seq=test_seq, zero_padding=True)
print seq_score

print len(seq_score)



