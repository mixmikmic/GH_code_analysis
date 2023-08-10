import pandas as pd
import os.path

BASE_PATH='/hackathon/Hackathon_Project_4/'
jp=lambda x: os.path.join(BASE_PATH,x)

pd.read_table(jp('metadata.tsv'))

metadata=pd.read_table(jp('metadata.tsv'))
metadata[:1]

metadata.columns

DEPTH_DIR='/hackathon/Hackathon_Project_4/ENCODE_DATA_GM12878/COMPLETED/DepthOfCoverage_test/'
jp2=lambda x: os.path.join(DEPTH_DIR,x)

pd.read_table(jp2('ENCFF000ATH_chr22_ChIP-seq_rg.depth.sample_gene_summary'))

pd.read_table(jp2('ENCFF000ATH_chr22_ChIP-seq_rg.depth.sample_interval_statistics'))

pd.read_table(jp2('ENCFF000ATH_chr22_ChIP-seq_rg.depth.sample_interval_summary'))

pd.read_table(jp2('ENCFF000ATH_chr22_ChIP-seq_rg.depth.sample_statistics'))

pd.read_table(jp2('ENCFF000ATH_chr22_ChIP-seq_rg.depth.sample_summary'))

get_ipython().run_cell_magic('bash', '', '\n#-l INFO --omitDepthOutputAtEachBase --omitLocusTable \\\ngatk -T DepthOfCoverage \\\n-R "/hackathon/Hackathon_Project_4/REFERENCE_GENOME/hg19_ENCODE.fa" \\\n-I "/hackathon/Hackathon_Project_4/ENCODE_DATA_GM12878/COMPLETED/rg_bams/ENCFF000ATH_chr22_ChIP-seq_rg.bam" \\\n-L chr22 \\\n--minBaseQuality 20 \\\n--minMappingQuality 10 \\\n--start 1 \\\n--stop 5000 \\\n--nBins 200 \\\n--includeRefNSites \\\n--countType COUNT_FRAGMENTS \\\n--calculateCoverageOverGenes "/hackathon/Hackathon_Project_4/REFERENCE_GENOME/refGene_chr22_sorted.txt" \\\n-o "/hackathon/Hackathon_Project_4/ENCODE_DATA_GM12878/COMPLETED/DepthOfCoverage_test/ENCFF000ATH_chr22_ChIP-seq_rg.depth"')

ENCFF000ATH_chr22_depth=pd.read_table(jp2('ENCFF000ATH_chr22_ChIP-seq_rg.depth'))
ENCFF000ATH_chr22_depth[:1]

ENCFF000ATH_chr22_depth[2:]

ENCFF000ATH_chr22_depth_moreThanTen = ENCFF000ATH_chr22_depth[ENCFF000ATH_chr22_depth['Total_Depth']>=10]

ENCFF000ATH_chr22_depth_moreThanTen[:3]

ENCFF000ATH_chr22_depth_moreThanTen[['Locus','Depth_for_20']]

get_ipython().run_cell_magic('bash', '', 'CHROMOSOME="chr22" # change this to any chromosome you like\nBASE_PATH=\'/hackathon/Hackathon_Project_4/\'\n\n#INDEXED_RG_BAMPATHS="/hackathon/Hackathon_Project_4/ENCODE_DATA_GM12878/COMPLETED/*_rg*bai" # TODO: change this to bam when everything is indexed\n#REFERENCE="/hackathon/Hackathon_Project_4/REFERENCE_GENOME/hg19_ENCODE.fa"\n#GENEFILE="/hackathon/Hackathon_Project_4/REFERENCE_GENOME/refGene_"$CHROMOSOME"_sorted.txt"\n#OUTPUT_DIRECTORY="/hackathon/Hackathon_Project_4/ENCODE_DATA_GM12878/COMPLETED/DepthOfCoverage"\n\nREFERENCE=$BASE_PATH"REFERENCE_GENOME/hg19_ENCODE.fa"\nGENEFILE=$BASE_PATH"REFERENCE_GENOME/refGene_"$CHROMOSOME"_sorted.txt"\n\nBAM_PATHS=$BASE_PATH"ENCODE_DATA_GM12878/COMPLETED/rg_bams/"\n#BAM_PATHS=$BASE_PATH"ENCODE_DATA_GM12878/COMPLETED/DepthOfCoverage/"\nOUTPUT_DIRECTORY=$BASE_PATH"ENCODE_DATA_GM12878/COMPLETED/DepthOfCoverage"\nBAMS=$BAM_PATHS"*_rg*bai" # Get indexed rg bam files (Run "readGroup.sh")\n#BAMS=$BAM_PATHS"ENCFF000WCN*_rg*bai" # Get indexed rg bam files (Run "readGroup.sh")\n#BAMS=$BAM_PATHS"ENCFF000WCQ*_rg*bai" # Get indexed rg bam files (Run "readGroup.sh")\n#BAMS=$BAM_PATHS"ENCFF000WD*_rg*bai" # Get indexed rg bam files (Run "readGroup.sh")\n#mkdir $OUTPUT_DIRECTORY\n\nDEPTH_COMMANDS_FILE=${OUTPUT_DIRECTORY}/"depth_commands.sh"\n\nfor bam in $(ls $BAMS);\ndo\n    # change the regex extension replacement \n    #DEPTH_COMMAND="gatk -T DepthOfCoverage -R $REFERENCE -I $(echo $bam|sed \'s/.bai//g\') -L $CHROMOSOME -l INFO --omitDepthOutputAtEachBase --omitLocusTable --minBaseQuality 20 --minMappingQuality 20 --start 1 --stop 5000 --nBins 200 --includeRefNSites --countType COUNT_FRAGMENTS --calculateCoverageOverGenes $GENEFILE -o ${OUTPUT_DIRECTORY}/$(basename $bam|sed \'s/.bam.bai/.depth/g\')"\n    DEPTH_COMMAND="gatk -T DepthOfCoverage -R $REFERENCE -I $(echo $bam|sed \'s/.bai//g\') -L $CHROMOSOME --minBaseQuality 20 --minMappingQuality 10 --start 1 --stop 5000 --nBins 200 --includeRefNSites --countType COUNT_FRAGMENTS --calculateCoverageOverGenes $GENEFILE -o ${OUTPUT_DIRECTORY}/$(basename $bam|sed \'s/.bam.bai/.depth/g\')"\n    $DEPTH_COMMAND\n    #echo $DEPTH_COMMAND\ndone >> $DEPTH_COMMANDS_FILE\n\n#head ${DEPTH_COMMANDS_FILE}')

get_ipython().run_cell_magic('bash', '', '\n#REFERENCE="/hackathon/Hackathon_Project_4/REFERENCE_GENOME/hg19.fa"\nbam="/hackathon/Hackathon_Project_4/VariantCall_HAPLOTYPE/merged_rg.chr22.bam" # TODO: change this to bam when everything is indexed\nCHROMOSOME="chr22" # change this to any chromosome you like\nBASE_PATH=\'/hackathon/Hackathon_Project_4/\'\nBAM_PATHS=$BASE_PATH"VariantCall_HAPLOTYPE/"\nOUTPUT_DIRECTORY=$BAM_PATHS"DepthOfCoverage"\nmkdir $OUTPUT_DIRECTORY\n\nREFERENCE=$BASE_PATH"REFERENCE_GENOME/hg19_ENCODE.fa"\nGENEFILE=$BASE_PATH"REFERENCE_GENOME/refGene_"$CHROMOSOME"_sorted.txt"\n\ngatk -T DepthOfCoverage -R $REFERENCE -I $(echo $bam|sed \'s/.bai//g\') -L $CHROMOSOME -l INFO --omitDepthOutputAtEachBase --omitLocusTable --minBaseQuality 20 --minMappingQuality 20 --start 1 --stop 5000 --nBins 200 --includeRefNSites --countType COUNT_FRAGMENTS --calculateCoverageOverGenes $GENEFILE -o ${OUTPUT_DIRECTORY}/$(basename $bam|sed \'s/.bam.bai/.depth/g\')')



