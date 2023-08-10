get_ipython().run_cell_magic('html', '', '<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">Welcome to my laboratory :)<br><br>Sequencing long ribosomal cluster from plants, insects &amp; fungi in real-time in the Amazon rainforest. Within a few mins of <a href="https://twitter.com/nanopore?ref_src=twsrc%5Etfw">@nanopore</a> data generated, performed BLAST &amp; got correct hits! Dual indexing looks great for pooling many samples<a href="https://twitter.com/hashtag/junglegenomics?src=hash&amp;ref_src=twsrc%5Etfw">#junglegenomics</a> <a href="https://t.co/UQVjYfmU8U">pic.twitter.com/UQVjYfmU8U</a></p>&mdash; Aaron Pomerantz (@AaronPomerantz) <a href="https://twitter.com/AaronPomerantz/status/980873273348038656?ref_src=twsrc%5Etfw">April 2, 2018</a></blockquote>\n<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>')

# dataDirectory is the path to our fast5 file.
# If you are using your own data, change dataDirectory to the path to your .fast5 files.
dataDirectory = 'data/'

# Print the number of fast5 files in the dataDirectory.
# Click the "Run" button at the top of this page to run this code.
get_ipython().system('find $dataDirectory -maxdepth 1 -name "*.fast5" | wc -l')

# The -q option stops poretools outputting any warning messages.
get_ipython().system('poretools stats -q $dataDirectory')

# Look at stats for forward strands
get_ipython().system('poretools stats -q --type fwd $dataDirectory')

# Look at stats for reverse strands
get_ipython().system('poretools stats -q --type rev $dataDirectory')

# Look at two-directional reads
get_ipython().system('poretools stats -q --type 2D $dataDirectory')

# Add squiggle plot here!!!

# Make a folder to store our fasta files in.
get_ipython().system('mkdir fastaOutput')

# Convert our fast5 files to fasta.
get_ipython().system('poretools fasta $dataDirectory > fastaOutput/outputPoretoolData.fasta')

# This will show us the first 200 characters of the first two lines of our file.
# We don't want to look at the whole sequence because it's going to be really long!
get_ipython().system('cut -c -200 fastaOutput/outputPoretoolData.fasta | head -2')

get_ipython().system('poretools winner $dataDirectory > winner.fasta')

get_ipython().system('wget https://nanopore.s3.climb.ac.uk/MAP006-1_2D_pass.fasta')

get_ipython().system('minimap2 -x ava-ont -t 1 MAP006-1_2D_pass.fasta MAP006-1_2D_pass.fasta > overlap.paf')

get_ipython().system('miniasm -f MAP006-1_2D_pass.fasta overlap.paf | awk \'/^S/{print ">"$2"\\n"$3}\' | fold > asm.fa')

# Retrieve genome sequences for each species
get_ipython().system('wget ftp://ftp.ensemblgenomes.org/pub/bacteria/release-33/fasta/bacteria_9_collection/escherichia_coli_0_1304/dna/Escherichia_coli_0_1304.GCA_000303235_1.dna.nonchromosomal.fa.gz')
get_ipython().system('wget ftp://ftp.ensemblgenomes.org/pub/release-38/bacteria//fasta/bacteria_15_collection/_clostridium_asparagiforme_dsm_15981/dna/_clostridium_asparagiforme_dsm_15981.ASM15807v1.dna.nonchromosomal.fa.gz')
get_ipython().system('wget ftp://ftp.ensemblgenomes.org/pub/release-38/bacteria//fasta/bacteria_176_collection/_bacillus_aminovorans/dna/_bacillus_aminovorans.ASM164324v1.dna.nonchromosomal.fa.gz')
get_ipython().system('wget https://downloads.yeastgenome.org/sequence/S288C_reference/orf_dna/orf_genomic_1000_all.fasta.gz ')
get_ipython().system('mv  orf_genomic_1000_all.fasta.gz yeast_genomic_1000_all.fa.gz ')

# The -o option allows us to name the sketch output file.
get_ipython().system('mash sketch *.fa.gz -o reference')

# Note: if there's no star on the left side of the code block ("In [*]:") then you know it's done running.
get_ipython().system('mash sketch -i asm.fa -o contigs')

get_ipython().system('mash dist reference.msh contigs.msh > distance.tab')

get_ipython().system('head distance.tab')

## A Shirt Code snippit that gets the closest Reference  for each Contig
import collections                                                             ##This line loads some extra python features for use
Reference = collections.namedtuple('Reference', ['name', 'distance','pvalue']) ##This creates a dummy "class/namespace" That has attributes for each componants of a distance measurement.
contigs   = collections.defaultdict(lambda : Reference(None, 1, 1)  )          ##A mapping between contigs and Reference_distances. By default each Contig is given the "null" reference.
with open("distance.tab")  as infile:                                          ##Opens a file for  reading this file is  a  list  of  lines.
    for reference, contig, distance, pvalue, matches in map(str.split,infile): ##this callsthe  string  split  function  on each line and  then upacks the splitline into 4 named columns.
        candidate = Reference(str(reference), float(distance), float(pvalue))  ##Creates a  Reference object for each contig  reference pair
        if contigs[contig].distance > candidate.distance:                      ##Compares each reference to the best reference.
                contigs[contig]  = candidate                                   ##Keeps the best  reference

##Print out the best Reference for each Contig
for contig, reference in contigs.items():            # Goes Through each contig  reference pair
    print("P-value   = ", reference.pvalue)
    print("Reference = ", reference.name)
    print("contig    = ",          contig)  # Prints the conig referenceand pvalue The best one!
    print()
##Can Print every Uniq taxa  found in the contig list!
# for taxa in set(reference.name  for reference  in  contigs.values()): #Iterates through every uniue taxa
#     print(taxa)                                                       #Prints out  each  unique taxa.

get_ipython().system('pwd')
get_ipython().system('ls /work/MetaGeneMark_linux_64/mgm')

get_ipython().system('gmhmmp -a -r -f G -d -m ../MetaGeneMark_linux_64/mgm/MetaGeneMark_v1.mod -o sequence.gff asm.fa')

get_ipython().system('head -20 sequence.gff')

get_ipython().system('head asm.fa')

get_ipython().system('sed -i -e 1,9d sequence.gff')

get_ipython().system('head -20 sequence.gff')
get_ipython().system('wc -l sequence.gff')

# This python script will get the start and stop indexes from the GFF 
# and get FASTA sequences from the assembly 

import csv

nameOfContig = list()
startIndexList = list()
stopIndexList = list()
# get start and stop indexes in the GFF file
with open("sequence.gff") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"): #You can also use delimiter="\t" rather than giving a dialect.
        if len(line) > 1:
            nameOfContig.append(""+str(line[2:3][0])+str(line[3:4][0])+"-"+str(line[4:5][0]))
            startIndexList.append(line[3:4])
            stopIndexList.append(line[4:5])
startAndStopList = list(zip(nameOfContig,startIndexList,stopIndexList))

# Use BioPython to assemble output FASTA file
from Bio import SeqIO
sequences = list()
for record in SeqIO.parse("asm.fa", "fasta"):
    print("This is the header for your assembly fasta: "+record.id)
    for name,start,stop in startAndStopList :
        if start != [] and stop != [] :
            sequences.append(record.seq[int(start[0]):int(stop[0])])
fastaList = list(zip(nameOfContig, sequences))
with open("annotatedGene.fa", "w") as output_handle:
    for name, seq in fastaList:
        fasta_format_string = ">"+name+"\n%s\n" % seq
        output_handle.write(fasta_format_string)

# Get the largest FASTA sequence
maxFasta = max(fastaList, key=lambda x: len(x[1]))
fasta_format_string = ">"+str(maxFasta[0])+"\n%s\n" % str(maxFasta[1])
print(fasta_format_string)

# Blastn the largest sequence
from Bio.Blast import NCBIWWW
result_handle = NCBIWWW.qblast("blastx", "nr", str(maxFasta[1]))




