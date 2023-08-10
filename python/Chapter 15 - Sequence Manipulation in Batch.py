get_ipython().system('curl https://raw.githubusercontent.com/Serulab/Py4Bio/master/samples/samples.tar.bz2 -o samples.tar.bz2')
get_ipython().system('mkdir samples')
get_ipython().system('tar xvfj samples.tar.bz2 -C samples')

get_ipython().system('conda install biopython -y')

import random

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

TOTAL_SEQUENCES = 500
MIN_SIZE = 400
MAX_SIZE = 1500

def new_rnd_seq(seq_len):
    """
    Generate a random DNA sequence with a sequence length
    of "sl" (int).
    return: A string with a DNA sequence.
    """
    s = ''
    while len(s) < seq_len:
        s += random.choice('ATCG')
    return s

with open('randomseqs.txt','w') as new_fh:
    for i in range(1, TOTAL_SEQUENCES + 1):
        # Select a random number between MIN_SIZE and MAX_SIZE
        rsl = random.randint(MIN_SIZE, MAX_SIZE)
        # Generate the random sequence
        rawseq = new_rnd_seq(rsl)
        # Generate a correlative name
        seqname = 'Sequence_number_{0}'.format(i)
        rec = SeqRecord(Seq(rawseq), id=seqname, description='')
        SeqIO.write([rec], new_fh, 'fasta')

from Bio import SeqIO

INPUT_FILE = 'samples/fasta22.fas'
OUTPUT_FILE = 'fasta22_out.fas'

def retseq(seq_fh):
    """
    Parse a fasta file and store non empty records
    into the fullseqs list.
    :seq_fh: File handle of the input sequence
    :return: A list with non empty sequences
    """
    fullseqs = []
    for record in SeqIO.parse(seq_fh,'fasta'):
        if len(record.seq):
            fullseqs.append(record)
    return fullseqs

with open(INPUT_FILE) as in_fh:
    with open(OUTPUT_FILE, 'w') as out_fh:
        SeqIO.write(retseq(in_fh), out_fh, 'fasta')

from Bio import SeqIO

INPUT_FILE = 'samples/fasta22.fas'
OUTPUT_FILE = 'fasta22_out2.fas'

def retseq(seq_fh):
    """
    Parse a fasta file and returns non empty records
    :seq_fh: File handle of the input sequence
    :return: Non empty sequences
    """
    for record in SeqIO.parse(seq_fh, 'fasta'):
        if len(record.seq):
            yield record

with open(INPUT_FILE) as in_fh:
    with open(OUTPUT_FILE, 'w') as out_fh:
        SeqIO.write(retseq(in_fh), out_fh, 'fasta')

from Bio import SeqIO

INPUT_FILE = 'fasta22_out.fas'
OUTPUT_FILE = 'fasta33.fas'

with open(INPUT_FILE) as in_fh:
    with open(OUTPUT_FILE, 'w') as out_fh:
        for record in SeqIO.parse(in_fh,'fasta'):
            if len(record.seq):
                SeqIO.write([record], out_fh, 'fasta')

from Bio import SeqIO

INPUT_FILE = 'fasta22_out.fas'
OUTPUT_FILE = 'fasta33.fas'

with open(INPUT_FILE) as in_fh:
    with open(OUTPUT_FILE, 'w') as out_fh:
        for record in SeqIO.parse(in_fh,'fasta'):
            # Modify description
            record.description += '[Rattus norvegicus]'
            SeqIO.write([record], out_fh, 'fasta')

INPUT_FILE = 'fasta22_out.fas'
OUTPUT_FILE = 'fasta33.fas'

with open(INPUT_FILE) as in_fh:
    with open(OUTPUT_FILE, 'w') as out_fh:
        for line in in_fh:
            if line.startswith('>'):
                line = line.replace('\n', '[Rattus norvegicus]\n')
            out_fh.write(line)

