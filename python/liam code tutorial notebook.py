# dataDirectory='~/Users/...'

dataDirectory='data/'

get_ipython().system('find $dataDirectory -maxdepth 1 -name "*.fast5" | wc -l')

get_ipython().system('poretools stats -q $dataDirectory')

get_ipython().system('poretools stats -q --type fwd $dataDirectory')

get_ipython().system('poretools stats -q --type rev $dataDirectory')

get_ipython().system('poretools stats -q --type 2D $dataDirectory')

#!poretools fasta $dataDirectory > fastaOutput/nameOfFile.fasta

get_ipython().system('mkdir fastaOutput')
get_ipython().system('poretools fasta $dataDirectory > fastaOutput/outputPoretoolData.fasta')

get_ipython().system('pwd')
get_ipython().system('ls /work/MetaGeneMark_linux_64/mgm')

get_ipython().system('gmhmmp -a -r -f G -d -m ../MetaGeneMark_linux_64/mgm/MetaGeneMark_v1.mod -o data/sequence.gff assembly.fa')

get_ipython().system('head -20 data/sequence.gff')

get_ipython().system('head data/asm.fa')

get_ipython().system('sed -i -e 1,9d data/sequence.gff')

get_ipython().system('head -20 data/sequence.gff')
get_ipython().system('wc -l data/sequence.gff')

# This python script will get the start and stop indexes from the GFF 
# and get FASTA sequences from the assembly 

import csv

nameOfContig = list()
startIndexList = list()
stopIndexList = list()
# get start and stop indexes in the GFF file
with open("data/sequence.gff") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"): #You can also use delimiter="\t" rather than giving a dialect.
        if len(line) > 1:
            nameOfContig.append(""+str(line[2:3][0])+str(line[3:4][0])+"-"+str(line[4:5][0]))
            startIndexList.append(line[3:4])
            stopIndexList.append(line[4:5])
startAndStopList = list(zip(nameOfContig,startIndexList,stopIndexList))

# Use BioPython to assemble output FASTA file
from Bio import SeqIO
sequences = list()
for record in SeqIO.parse("data/asm.fa", "fasta"):
    print("This is the header for your assembly fasta: "+record.id)
    for name,start,stop in startAndStopList :
        if start != [] and stop != [] :
            sequences.append(record.seq[int(start[0]):int(stop[0])])
fastaList = list(zip(nameOfContig, sequences))
with open("data/annotatedGene.fa", "w") as output_handle:
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


