get_ipython().system('curl https://s3.amazonaws.com/py4bio/TAIR.tar.bz2 -o TAIR.tar.bz2')
get_ipython().system('mkdir samples')
get_ipython().system('tar xvfj TAIR.tar.bz2 -C samples')
get_ipython().system('curl https://s3.amazonaws.com/py4bio/ncbi-blast-2.6.0.tar.bz2 -o ncbi-blast-2.6.0.tar.bz2')
get_ipython().system('tar xvfj ncbi-blast-2.6.0.tar.bz2')
get_ipython().system('curl https://s3.amazonaws.com/py4bio/clustalw2 -o clustalw2')
get_ipython().system('chmod a+x clustalw2')
get_ipython().system('ls')

get_ipython().system('conda install biopython -y')

import sqlite3
from Bio import SeqIO

SEQ_FILE = open('samples/TAIR10_seq_20101214_updated')
CDS_FILE = open('samples/TAIR10_cds_20101214_updated')
AT_DB_FILE = 'AT.db'

at_d = {}
# Get all sequences from TAIR sequences file.
for record in SeqIO.parse(SEQ_FILE, 'fasta'):
    sid = record.id
    seq = str(record.seq)
    at_d[sid] = [seq]
# Get all sequences from TAIR CDS file.
for record in SeqIO.parse(CDS_FILE, 'fasta'):
    sid = record.id
    seq = str(record.seq)
    at_d[sid].append(seq)
# Write to a CSV file only the entries of the dictionary that
# has data from both sources
conn = sqlite3.connect(AT_DB_FILE)
c = conn.cursor()
c.execute('create table seq(id, cds, full_seq)')
for seq_id in at_d:
    if len(at_d[seq_id])==2:
        # Write in this order: ID, CDS, FULL_SEQ.
        c.execute('INSERT INTO seq VALUES (?,?,?)',
                 ((seq_id, at_d[seq_id][1], at_d[seq_id][0])))
conn.commit()
conn.close()

#!/usr/bin/env python

import os
import sqlite3

from Bio import SeqIO, SeqRecord, Seq
from Bio.Align.Applications import ClustalwCommandline
from Bio.Blast import NCBIXML
from Bio.Blast.Applications import NcbiblastnCommandline as bn
from Bio import AlignIO

AT_DB_FILE = 'AT.db'
BLAST_EXE = 'ncbi-blast-2.6.0+/bin/blastn'
BLAST_DB = 'ncbi-blast-2.6.0+/db/TAIR10'
CLUSTALW_EXE = os.path.join(os.getcwd(), 'clustalw2')

input_sequence = """>XM_013747562.1 PREDICTED: Brassica oleracea var. oleracea
ATCTTTCGCGAGAGGTTCATTATTGTCCGGAAGAGGTGCTCATGTTTTGGTAAAGCGATCACAAGGTGTT
CGATACAATACCTGAGAGAGTTTCCACAGCTTTCTTCTGATTCTTACTCGGTTTGAGTGAGCTGGATCTT
CCACGACGAAGATGATGATCTTGGATGTTTGCAATGAGATTATAAAGATCCAGAAGCTAAGACGGGTTGT
CTCTTACGCTGGATTCTACTGCTTCACTGCAGCCCTCACATTCTTCTACACAAACAACACAACAAGAGCA
GGATTCTCCAGGGGAGATCAGTTTTATGCGTCTTACCCTGCGGGTACCGAACTTCTTACCGACACAGCTA
AGCTGTACAAAGCGGCGCTTGGGAATTGCTATGAATCTGAGGATTGGGGTCCTGTCGAGTTCTGCATAAT
GGCTAAGCATTTTGAGCGCCAGGGAAAGTCTCCATACGTTTACCACTCTCAATACATGGCTCACCTTCTT
TCACAAGGCCAACTTGATGGAAGTGGCTAGAGTCGTTGATGACTTGCAAGACAGCTCCTTTTTCAATCTG
TGTACCTAATCTTGTTATTGGAACTTCCTTCTTTACTCTTTTTCCGAATTTGTACGGCGATGGTATTTGA
GGTTACCACCAAGAAATATAAGAACATGTTCTGGTGTAGACAATGAATGTAATAAACACATAAGATCAGA
CCTTGATATGA
"""

with open('input_sequence.fasta', 'w') as in_seq:
    in_seq.write(input_sequence)

def allgaps(seq):
    """Return a list with tuples containing all gap positions
       and length. seq is a string."""
    gaps = []
    indash = False
    for i, c in enumerate(seq):
        if indash is False and c == '-':
            c_ini = i
            indash = True
            dashn = 0
        elif indash is True and c == '-':
            dashn += 1
        elif indash is True and c != '-':
            indash = False
            gaps.append((c_ini, dashn+1))
    return gaps

def iss(user_seq):
    """Infer Splicing Sites from a FASTA file full of EST
    sequences"""

    with open('forblast','w') as forblastfh:
        forblastfh.write(str(user_seq.seq))

    blastn_cline = bn(cmd=BLAST_EXE, query='forblast',
                      db=BLAST_DB, evalue='1e-10', outfmt=5,
                      num_descriptions='1',
                      num_alignments='1', out='outfile.xml')
    blastn_cline()
    b_record = NCBIXML.read(open('outfile.xml'))
    title = b_record.alignments[0].title
    sid = title[title.index(' ')+1 : title.index(' |')]
    # Polarity information of returned sequence.
    # 1 = normal, -1 = reverse.
    frame = b_record.alignments[0].hsps[0].frame[1]
    # Run the SQLite query
    conn = sqlite3.connect(AT_DB_FILE)
    c = conn.cursor()
    res_cur = c.execute('SELECT CDS, FULL_SEQ from seq '
                        'WHERE ID=?', (sid,))
    cds, full_seq = res_cur.fetchone()
    if cds=='':
        print('There is no matching CDS')
        exit()
    # Check sequence polarity.
    sidcds = '{0}-CDS'.format(sid)
    sidseq = '{0}-SEQ'.format(sid)
    if frame==1:
        seqCDS = SeqRecord.SeqRecord(Seq.Seq(cds),
                                     id = sidcds,
                                     name = '',
                                     description = '')
        fullseq = SeqRecord.SeqRecord(Seq.Seq(full_seq),
                                      id = sidseq,
                                      name='',
                                      description='')
    else:
        seqCDS = SeqRecord.SeqRecord(
            Seq.Seq(cds).reverse_complement(),
            id = sidcds, name='', description='')
        fullseq = SeqRecord.SeqRecord(
            Seq.Seq(full_seq).reverse_complement(),
            id = sidseq, name = '', description='')
    # A tuple with the user sequence and both AT sequences
    allseqs = (record, seqCDS, fullseq)
    with open('foralig.txt','w') as trifh:
        # Write the file with the three sequences
        SeqIO.write(allseqs, trifh, 'fasta')
    # Do the alignment:
    outfilename = '{0}.aln'.format(user_seq.id)
    cline = ClustalwCommandline(CLUSTALW_EXE,
                                infile = 'foralig.txt',
                                outfile = outfilename,
                                )
    cline()
    # Walk over all sequences and look for query sequence
    for seq in AlignIO.read(outfilename, 'clustal'):
        if user_seq.id in seq.id:
            seqstr = str(seq.seq)
            gaps = allgaps(seqstr.strip('-'))
            break
    print('Original sequence: {0}'.format(user_seq.id))
    print('\nBest match in AT CDS: {0}'.format(sid))
    acc = 0
    for i, gap in enumerate(gaps):
        print('Putative intron #{0}: Start at position {1}, '
              'length {2}'.format(i+1, gap[0]-acc, gap[1]))
        acc += gap[1]
    print('\n{0}'.format(seqstr.strip('-')))
    print('\nAlignment file: {0}\n'.format(outfilename))

description = 'Program to infer intron position based on '               'Arabidopsis Thaliana genome'
with open('input_sequence.fasta', 'r') as seqhandle:
    records = SeqIO.parse(seqhandle, 'fasta')
    for record in records:
        iss(record)

