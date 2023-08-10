import os
from Bio import SeqIO 
import shutil
import subprocess

#Define the PATH for variable parameters
BASE_AA_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/Pst_104E_v12'
BASE_A_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/032017_assembly'
OUT_PATH_NUCMER = os.path.join(BASE_AA_PATH, 'nucmer_analysis')
OUT_PATH_ASSEMBLETICS = os.path.join(BASE_AA_PATH, 'Assembletics')
MUMMER_PATH_PREFIX = '/home/benjamin/anaconda3/bin/'
ASSEMBLETICS_PATH = '/home/benjamin/genome_assembly/PST79/FALCON/p_assemblies/v9_1/downstream_analysis_2017/scripts/Assemblytics'
py2_env = 'py27' #python 2 environment in conda 
if not os.path.isdir(OUT_PATH_NUCMER):
    os.mkdir(OUT_PATH_NUCMER)
if not os.path.isdir(OUT_PATH_ASSEMBLETICS):
    os.mkdir(OUT_PATH_ASSEMBLETICS)

#Define your p and h genome and move it into the allele analysis folder
genome_prefix  = 'Pst_104E_v12'
p_genome = 'Pst_104E_v12_p_ctg'
h_genome = 'Pst_104E_v12_h_ctg'
genome_file_suffix = '.genome_file'
for x in (x + '.fa' for x in [p_genome, h_genome]):
    shutil.copy2(BASE_A_PATH+'/'+x, OUT_PATH_NUCMER)
    shutil.copy2(BASE_A_PATH+'/'+x, OUT_PATH_ASSEMBLETICS)

#define the scripts to generate
bash_script_q= genome_prefix+"_ph_ctg_qmapping.sh"
bash_script_g=genome_prefix+"_ph_ctg_gmapping.sh"
bash_script_nucmer_assemlytics = genome_prefix +"_ncumer_assembleticsmapping.sh"
bash_script_assemlytics = genome_prefix + '_assembletics.sh'

outfq = open(os.path.join(OUT_PATH_NUCMER, bash_script_q), 'w')
outfq.write('#!/bin/bash\n')
outfg = open(os.path.join(OUT_PATH_NUCMER,bash_script_g), 'w')
outfg.write('#!/bin/bash\n')#parsing out P and corresponding h contigs and writing a short ncumer script that aligns them against each other
outfna = open(os.path.join(OUT_PATH_ASSEMBLETICS,bash_script_nucmer_assemlytics), 'w')
outfna.write('#!/bin/bash\n')

for pseq_record in SeqIO.parse(OUT_PATH_ASSEMBLETICS+'/'+genome_prefix+'_p_ctg.fa', 'fasta'):
    p_acontigs = []
    p_contig = pseq_record.id.split("_")[0]+"_"+pseq_record.id.split("_")[1]
    suffix = genome_prefix+"_"+p_contig+"_php"
    p_file = genome_prefix+"_"+p_contig+'.fa'
    SeqIO.write(pseq_record, OUT_PATH_ASSEMBLETICS+'/'+ p_file, 'fasta')
    SeqIO.write(pseq_record,OUT_PATH_NUCMER+'/'+ p_file, 'fasta')
    for aseq_record in SeqIO.parse(OUT_PATH_ASSEMBLETICS+'/'+genome_prefix+'_h_ctg.fa', 'fasta'):
        if aseq_record.id.split("_")[1]  == pseq_record.id.split("_")[1]:
            p_acontigs.append(aseq_record)
    a_file = genome_prefix +"_"+pseq_record.id.split("_")[0]+"_"+pseq_record.id.split("_")[1]+'_h_ctgs.fa'
    #if we have alternative contigs safe those too
    if p_acontigs != []:
        SeqIO.write(p_acontigs, OUT_PATH_ASSEMBLETICS+'/'+  a_file, 'fasta')
        SeqIO.write(p_acontigs, OUT_PATH_NUCMER+'/'+  a_file, 'fasta')
        outfq.write(MUMMER_PATH_PREFIX +'/nucmer '+p_file+' '+a_file+" > "+'out.delta\n')
        outfq.write(MUMMER_PATH_PREFIX +'/delta-filter -q '+'out.delta'+" > "+suffix+"_qfiltered.delta\n")
        outfq.write(MUMMER_PATH_PREFIX +'/show-coords -T '+suffix+"_qfiltered.delta > "+suffix+".qcoords\n")
        outfq.write(MUMMER_PATH_PREFIX +'/mummerplot -p '+suffix+'_qfiltered --png '+suffix+"_qfiltered.delta\n")
        outfq.write(MUMMER_PATH_PREFIX +'/mummerplot -c -p '+suffix+'_qfiltered_cov --png '+suffix+"_qfiltered.delta\n")
        #for g_file bash script
        outfg.write(MUMMER_PATH_PREFIX +'/nucmer '+p_file+' '+a_file+" > "+'out.delta\n')
        outfg.write(MUMMER_PATH_PREFIX +'/delta-filter -g '+'out.delta'+" > "+suffix+"_gfiltered.delta\n")
        outfg.write(MUMMER_PATH_PREFIX +'/show-coords -T '+suffix+"_gfiltered.delta > "+suffix+".gcoords\n")
        outfg.write(MUMMER_PATH_PREFIX +'/mummerplot -p '+suffix+'_gfiltered --png '+suffix+"_gfiltered.delta\n")
        outfg.write(MUMMER_PATH_PREFIX +'/mummerplot -c -p  '+suffix+'_gfiltered_cov --png '+suffix+"_gfiltered.delta\n")
        #for nucmer assemblytics out
        outfna.write(MUMMER_PATH_PREFIX +'/nucmer -maxmatch -l 100 -c 500 '+p_file+' '+a_file+' -prefix ' + suffix +'\n')
outfna.close()    
outfq.close()
outfg.close()  

#run the scripts and check if they are errors
bash_script_q_stderr = subprocess.check_output('bash %s' %os.path.join(OUT_PATH_NUCMER, bash_script_q), shell=True, stderr=subprocess.STDOUT)
bash_script_g_stderr = subprocess.check_output('bash %s' %os.path.join(OUT_PATH_NUCMER, bash_script_g), shell=True, stdeorr=subprocess.STDOUT)
bash_script_assembletics_stderr = subprocess.check_output('bash %s' %os.path.join(OUT_PATH_ASSEMBLETICS, bash_script_nucmer_assemlytics), shell=True, stderr=subprocess.STDOUT)

bash_script_assembletics_stderr = subprocess.check_output('bash %s' %os.path.join(OUT_PATH_ASSEMBLETICS, bash_script_nucmer_assemlytics), shell=True, stderr=subprocess.STDOUT)

#write the Assemblytics script
outfnarun = open(os.path.join(OUT_PATH_ASSEMBLETICS, bash_script_assemlytics), 'w')
outfnarun.write('#!/bin/bash\n')
delta_files = [x for x in os.listdir(OUT_PATH_ASSEMBLETICS) if x.endswith('delta')]
outfnarun.write('export PATH="%s":$PATH\n'% ASSEMBLETICS_PATH)
outfnarun.write('#Assemblytics delta_file output_prefix unique_anchor_length maximum_feature_length path_to_R_scripts\n')
outfnarun.write('source activate %s\n' %py2_env)
for delta in delta_files:
    folder_name = delta[:-6] + '_8kbp'
    output_prefix = delta[:-6] + '_8kbp_50kp'
    outfnarun.write("mkdir %s\ncp %s %s\ncd %s\n" % (folder_name, delta, folder_name, folder_name))
    outfnarun.write("Assemblytics %s %s 8000 50000 %s\n" % (delta, output_prefix, ASSEMBLETICS_PATH) )
    output_prefix = delta[:-6] + '_8kbp_10kp'
    outfnarun.write("Assemblytics %s %s 8000 10000 %s\ncd ..\n" % (delta, output_prefix, ASSEMBLETICS_PATH) )
    
outfnarun.write('source deactivate\n')
outfnarun.close()

bash_script_assemlytics_stderr = subprocess.check_output('bash %s' %os.path.join(OUT_PATH_ASSEMBLETICS, bash_script_assemlytics), shell=True, stderr=subprocess.STDOUT)

