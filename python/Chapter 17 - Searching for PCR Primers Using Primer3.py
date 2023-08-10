get_ipython().system('curl https://raw.githubusercontent.com/Serulab/Py4Bio/master/samples/samples.tar.bz2 -o samples.tar.bz2')
get_ipython().system('mkdir samples')
get_ipython().system('tar xvfj samples.tar.bz2 -C samples')

get_ipython().system('conda install biopython -y')

from Bio import SeqIO

sfile = open('samples/hsc1.fasta')
# mysel stores a SeqRecord object generated from the
# first record in the fasta file.
myseq = SeqIO.read(sfile, "fasta")
# title stores the "id" attribute of the SeqRecord object.
title = myseq.id
seq = str(myseq.seq).upper()
win_size = 45
i = 0
number_l = []
# This while is used to walk over the sequence.
while i<=(len(seq)-win_size):
    # Each position of number_l stores the amount of 'AAT'
    # found on each window.
    number_l.append(seq[i:i + win_size].count('AAT'))
    i += 1 # This is the same as i = i+1
# pos stores the position of the window with the highest
# amount of 'AAT'
pos = number_l.index(max(number_l))
data = {'title': title, 'seq': seq, 'pos': pos, 'win_size':
        win_size, 'len_seq': len(seq)}
# Saves the data formated as the input file needed by
# primer3.
with open('swforprimer3.txt','w') as f_out:
    with open('template') as tpl:
        completed = tpl.read().format(**data)
        f_out.write(completed)

from Bio import SeqIO
from Bio.Emboss.Applications import Primer3Commandline

INPUT_SEQUENCE = open('samples/hsc1.fasta')
OUTPUT_SEQUENCE = 'primer.txt'
sfile = open('samples/hsc1.fasta')
myseq = SeqIO.read(sfile, 'fasta')
title = myseq.id
seq = str(myseq.seq).upper()
win_size = 45
i = 0
number_l = []
while i <= (len(seq) - win_size):
    number_l.append(seq[i:i + win_size].count('AAT'))
    i += 1 # This is the same as i = i+1
pos = number_l.index(max(number_l))
pr_cl = Primer3Commandline(sequence=INPUT_SEQUENCE, auto=True)
pr_cl.outfile = OUTPUT_SEQUENCE
pr_cl.osize = 18
pr_cl.maxsize = 20
pr_cl.minsize = 15
pr_cl.explainflag = 1
pr_cl.target = (pos, win_size)
pr_cl.prange = (win_size, len(seq))
primer_cl()

