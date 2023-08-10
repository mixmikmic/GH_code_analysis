# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tnrange, tqdm_notebook
sys.path.insert(0, '/home/bay001/projects/codebase/rbp-maps/maps/')
from density import Map
from density import ReadDensity
from density import normalization_functions
from density import RDPlotter


colors = sns.color_palette('hls',14)

out_dir = '/home/bay001/projects/encode/analysis/conservation_analysis/conservation_plots/'

def create_tmp_region_bed_files(bed):
    """
    create temp bedfiles for each region
    """
    try:
        dx = pd.read_table(bed, names=['chrom','start','stop','l10p','l2fc','strand','priority','annotation'])
        dx['region'] = dx.apply(extract_priority, axis=1)
        regions = ['3UTR','5UTR','intron','CDS','noncoding_intron','noncoding_exon']

        for region in regions:
            df = dx[dx['region'].str.contains(region)]
            df['mid'] = ((df['stop'] + df['start'])/2).astype(int)
            df['mid1'] = df['mid'] + 1
            df = df[['chrom','mid','mid1','l10p','l2fc','strand']]
            # print(bed, region)
            df.to_csv(bed.replace('.bed','.{}.bed'.format(region)), sep='\t', header=False, index=False)
    except ValueError as e:
        print(e, bed)
        
def extract_priority(row):
    """
    from an annotation string, return the region with the highest priority
    """
    if 'INTERGENIC' not in row['priority']:
        return row['priority'].split(':')[4]
    return 'intergenic'

def plot_heatmap(phastcon, wd, out_dir, bed_prefix):
    
    i=0
    beds = glob.glob(os.path.join(wd,'{}*.BED'.format(bed_prefix)))
    if len(beds) != 4:
        print("warning: only {} beds found for prefix {}".format(len(beds), bed_prefix))
    for bed in sorted(beds):
        # fig = plt.gcf()
        fig, ax = plt.subplots(figsize=(20, 20))
        # fig.set_size_inches(20, 20)
        output_filename = os.path.join(out_dir, os.path.basename(bed).replace('.BED','.heatmap.png'))
        annotation = {bed:'bed'}
        phast = ReadDensity.Phastcon(phastcon)
        m = Map.Map(
            ip=phast,
            annotation=annotation,
            output_filename=output_filename,
            norm_function=normalization_functions.get_density,
            upstream_offset=300,downstream_offset=300,
            min_density_threshold=0, is_scaled=True, conf=1
        )

        m.create_matrix()
        if m.raw_matrices['ip'][bed].shape[0] > 1: # if we have more than 1 event to cluster...
            sns.heatmap(
                m.raw_matrices['ip'][bed],
                yticklabels=False, 
                xticklabels=False,
            )
            # plt.title(bed_prefix)
            # plt.xlabel("position (-100, center, +100)")
            # plt.ylabel("peak coordinates")
            # plt.tight_layout()
            plt.savefig(os.path.join(wd, output_filename))
            m.raw_matrices['ip'][bed].to_csv(os.path.join(wd, output_filename.replace('.png','.txt')))
            plt.clf()
            plt.cla()

    
def plot_region(phastcon, wd, out_dir, bed_prefix, kind='mean'):
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    i=0
    beds = glob.glob(os.path.join(wd,'{}*.BED'.format(bed_prefix)))
    # print("bed prefix: {}".format(bed_prefix))
    # print("beds found for mean plot: {}".format(beds))
    if len(beds) != 4:
        print("warning: only {} beds found for prefix {}".format(len(beds), bed_prefix))
    for bed in sorted(beds):
        output_filename = os.path.join(out_dir, os.path.basename(bed).replace('.BED','.tmp'))
        annotation = {bed:'bed'}
        phast = ReadDensity.Phastcon(phastcon)
        m = Map.Map(
            ip=phast,
            annotation=annotation,
            output_filename=output_filename,
            norm_function=normalization_functions.get_density,
            upstream_offset=300,downstream_offset=300,
            min_density_threshold=0, is_scaled=True, conf=1
        )

        m.create_matrix()
        if kind=='mean':
            line = m.raw_matrices['ip'][bed].mean()
        elif kind=='median':
            line = m.raw_matrices['ip'][bed].median()
        else:
            print("Invalid.")
            return 1
        
        x = plt.plot(
            line, 
            label=os.path.basename(bed).replace(bed_prefix + '.','') + "({})".format(m.raw_matrices['ip'][bed].shape[0]),
            color=colors[i]
        )
        i+=1
    plt.title(bed_prefix)
    plt.legend(loc=1)
    # print(bed_prefix + '.png')
    plt.savefig(os.path.join(out_dir, bed_prefix + '.{}.png'.format(kind)))
    plt.clf()
    plt.cla()

import glob
phastcon = '/projects/ps-yeolab/genomes/hg19/hg19.100way.phastCons.bw'
# peakdir = '/projects/ps-yeolab3/encode/analysis/encode_idr_clip_analysis/assigned'
peakdir = '/home/bay001/projects/encode/analysis/conservation_analysis/idr_peaks/'
wd = '/home/bay001/projects/encode/analysis/conservation_analysis/assigned_random_regions/'
annotated_files = glob.glob(os.path.join(peakdir,'*IDR*annotated*.real.BED'))
len(annotated_files)

file_prefixes = []
for file_name in annotated_files:
    file_prefixes.append(os.path.basename(file_name).split('.clip_formatted.bed')[0] + ".clip_formatted.bed")
file_prefixes = list(set(file_prefixes)) # get all prefixes 
len(file_prefixes)
file_prefixes[:5]

out_dir = '/home/bay001/projects/encode/analysis/conservation_analysis/conservation_plots'

progress = tnrange(len(file_prefixes))
for prefix in file_prefixes:
    for region in ['three_prime_utrs','five_prime_utrs','proxintron500','distintron500','cds']:
        region_prefix = prefix + '.' + region
        # print("region prefix: {}".format(region_prefix))
        plot_region(phastcon, peakdir, out_dir, region_prefix, kind='mean')
        plot_heatmap(phastcon, peakdir, out_dir, region_prefix)
    progress.update(1)









