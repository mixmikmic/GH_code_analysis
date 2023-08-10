get_ipython().system('curl https://raw.githubusercontent.com/Serulab/Py4Bio/master/samples/samples.tar.bz2 -o samples.tar.bz2')
get_ipython().system('mkdir samples')
get_ipython().system('tar xvfj samples.tar.bz2 -C samples')

from Bio import SeqIO, SeqRecord, Seq
from Bio.Alphabet import IUPAC

GB_FILE = 'samples/NC_006581.gb'
OUT_FILE = 'nadh.fasta'
with open(GB_FILE) as gb_fh:
    record = SeqIO.read(gb_fh, 'genbank')
seqs_for_fasta = []
for feature in record.features:
    # Each Genbank record may have several features, the program
    # will walk over all of them.
    qualifier = feature.qualifiers
    # Each feature has several parameters
    # Pick selected parameters.
    if 'NADH' in qualifier.get('product',[''])[0] and     'product' in qualifier and 'translation' in qualifier:
        id_ = qualifier['db_xref'][0][3:]
        desc = qualifier['product'][0]
        # nadh_sq is a NADH protein sequence
        nadh_sq = Seq.Seq(qualifier['translation'][0], IUPAC.protein)
        # 'srec' is a SeqRecord object from nadh_sq sequence.
        srec = SeqRecord.SeqRecord(nadh_sq, id=id_, description=desc)
        # Add this SeqRecord object into seqsforfasta list.
        seqs_for_fasta.append(srec)
with open(OUT_FILE, 'w') as outf:
    # Write all the sequences as a FASTA file.
    SeqIO.write(seqs_for_fasta, outf, 'fasta')

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

GB_FILE = 'samples/NC_006581.gb'
OUT_FILE = 'tg.fasta'
with open(GB_FILE) as gb_fh:
    record = SeqIO.read(gb_fh, 'genbank')
seqs_for_fasta = []
tg = (['cox2'],['atp6'],['atp9'],['cob'])
for feature in record.features:
    if feature.qualifiers.get('gene') in tg and feature.type=='gene':
        # Get the name of the gene
        genename = feature.qualifiers.get('gene')
        # Get the start position
        startpos = feature.location.start.position
        # Get the required slice
        newfrag = record.seq[startpos-1000: startpos]
        # Build a SeqRecord object
        newrec = SeqRecord(newfrag, genename[0] + ' 1000bp upstream',
                           '','')
        seqs_for_fasta.append(newrec)
with open(OUT_FILE,'w') as outf:
    # Write all the sequences as a FASTA file.
    SeqIO.write(seqs_for_fasta, outf, 'fasta')

