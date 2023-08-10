import os

get_ipython().system('apt-get install sra-toolkit')

get_ipython().system('mkdir ../fastq')
os.chdir('../fastq')

#A3SS_DNA:
get_ipython().system('wget ftp://ftp-trace.ncbi.nlm.nih.gov/sra/sra-instant/reads/ByExp/sra/SRX%2FSRX135%2FSRX1353954/SRR2723893/SRR2723893.sra')
get_ipython().system('fastq-dump SRR2723893.sra --split-files')
get_ipython().system('mv SRR2723893_1.fastq A3SS_dna_R1.fq')
get_ipython().system('mv SRR2723893_2.fastq A3SS_dna_R2.fq')

#A5SS_DNA:
get_ipython().system('wget ftp://ftp-trace.ncbi.nlm.nih.gov/sra/sra-instant/reads/ByExp/sra/SRX%2FSRX135%2FSRX1353956/SRR2723896/SRR2723896.sra')
get_ipython().system('fastq-dump SRR2723896.sra --split-files')
get_ipython().system('mv SRR2723896_1.fastq A5SS_dna_R1.fq')
get_ipython().system('mv SRR2723896_2.fastq A5SS_dna_R2.fq')

#A3SS_RNA:
get_ipython().system('wget ftp://ftp-trace.ncbi.nlm.nih.gov/sra/sra-instant/reads/ByExp/sra/SRX%2FSRX135%2FSRX1353955/SRR2723895/SRR2723895.sra')
get_ipython().system('fastq-dump SRR2723895.sra --split-files')
get_ipython().system('mv SRR2723895_1.fastq A3SS_rna_R1.fq')
get_ipython().system('mv SRR2723895_2.fastq A3SS_rna_R2.fq')
#A5SS_RNA:
get_ipython().system('wget ftp://ftp-trace.ncbi.nlm.nih.gov/sra/sra-instant/reads/ByExp/sra/SRX%2FSRX135%2FSRX1353957/SRR2723898/SRR2723898.sra')
get_ipython().system('fastq-dump SRR2723898.sra --split-files')
get_ipython().system('mv SRR2723898_1.fastq A5SS_rna_R1.fq')
get_ipython().system('mv SRR2723898_2.fastq A5SS_rna_R2.fq')

