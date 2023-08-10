get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
from rnaseq import rmats_inclevel_analysis as rmats
from encode import manifest_helpers
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tnrange, tqdm_notebook
import pybedtools

pd.set_option('display.max_columns', 500)

exprs = '/home/elvannostrand/data/clip/CLIPseq_analysis/RNAseq_expression/HepG2K562wnoncoding_ensg_onecelltypeonly.txt'
wd = '/home/bay001/projects/encode/analysis/cell_specific_overlaps'
splice_dir = '/projects/ps-yeolab3/bay001/maps/current_normed_annotations/se/'

# hepg2 = '/projects/ps-yeolab3/bay001/maps/current_normed_annotations/se/U2AF1-BGHLV30-HepG2-SE.MATS.JunctionCountOnly.significant.txt'
# k562 = '/projects/ps-yeolab3/bay001/maps/current_normed_annotations/se/U2AF1-LV08-K562-SE.MATS.JunctionCountOnly.significant.txt'
hepg2 = '/projects/ps-yeolab3/bay001/maps/current_normed_annotations/se/U2AF2-BGHLV26-HepG2-SE.MATS.JunctionCountOnly.significant.txt'
k562 = '/projects/ps-yeolab3/bay001/maps/current_normed_annotations/se/U2AF2-LV08-K562-SE.MATS.JunctionCountOnly.significant.txt'


import pybedtools

def rmats2bedtool_all(annotations, exon_start, exon_end):
    """
    Returns merged intervals that contain all regions from all 
    junction counts files. This ensures that regions will be non-overlapping
    and can be used as 'keys' to regions on the epigenome browser.
    """
    intervals = []
    # progress = tnrange(len(annotations))
    for pos in annotations:
        nice_name = os.path.basename(pos)
        nice_name = nice_name.replace('-SE.MATS.JunctionCountOnly','')
        nice_name = nice_name.replace('nr.txt','')
        # nice_name = os.path.basename(pos).replace('-SE.MATS.JunctionCountOnly.positive.nr.txt','')
        # nice_name = nice_name.replace('-SE.MATS.JunctionCountOnly.negative.nr.txt','')
        
        df = pd.read_table(pos)
        for col, row in df.iterrows():
            intervals.append(
                pybedtools.create_interval_from_list(
                    [row['chr'], str(row[exon_start]), str(row[exon_end]), nice_name, str(row['IncLevelDifference']), row['strand']])
            )
        # progress.update(1)
    bedtool_all_intervals = pybedtools.BedTool(intervals)
    return bedtool_all_intervals.sort().merge()

def rmats2bedtool(annotation, exon_start, exon_end):
    """
    Returns a bedtool object from an rmats annotation
    Uses the exonStart_0base and exonEnd as coordinates.
    """
    intervals = []
    df = pd.read_table(annotation)
    # progress = tnrange(df.shape[0], leave=False)
    for _, row in df.iterrows():
        intervals.append(
            pybedtools.create_interval_from_list(
                [
                    row['chr'], str(row[exon_start]), 
                    str(row[exon_end]), row['GeneID'], 
                    str(row['IncLevelDifference']), row['strand']
                ]
            )
        )
        # progress.update(1)
    
    return pybedtools.BedTool(intervals)

def transform_individual_rmats_positions(rmats_file, big_merged_bedtool, es, ee):
    """
    turns individual rmats exon positions into something 
    common to those found in the big merged bedtool file.
    
    Parameters:
    rmats_file : string
        rmats JunctionCountsOnly.txt
    big_merged_bedtool : pybedtools.BedTool
        intervals containing all regions for all annotations being compared.
    
    Returns: dataframe
    """
    
    # just get the "nice name" (RBP_CELL without the extra stuff)
    nice_name = os.path.basename(rmats_file)
    
    # nice_name = nice_name.replace('-SE.MATS.JunctionCountOnly..nr.txt','')
    # nice_name = nice_name.replace('-SE.MATS.JunctionCountOnly.negative.nr.txt','')
    nice_name = nice_name.split('-')
    nice_name = '{}_{}'.format(nice_name[0], nice_name[2])
    
    # for each rmats file, intersect with the merged bedtool to bin regions into those that are common amongst all
    individual_rmats_bedtool = rmats2bedtool(rmats_file, es, ee).sort()
    intersected = individual_rmats_bedtool.intersect(big_merged_bedtool, wb=True).to_dataframe()
    
    # thickStart, thickEnd, itemRgb actually contain the 'key' common regions from the big_merged_bedtool intersection.
    # intersected['chrom'] = intersected['thickStart']
    # intersected['start'] = intersected['thickEnd']
    # intersected['end'] = intersected['itemRgb']
    
    # re-format so that it's a proper dataframe, and re-name the 'score' column to be that of the name of the RBP. 
    # intersected = intersected[['chrom','start','end','name','score','strand']]
    intersected.columns = ['chrom','start','end','name','{}'.format(nice_name),'strand', 
                           'original_chr_{}'.format(nice_name), 'original_start_{}'.format(nice_name), 'original_end_{}'.format(nice_name)]
    intersected.set_index(['chrom','start','end','name','strand'], inplace=True)
    return intersected

def merge_all_rmats_transformed(all_annotations, big_merged_bedtool):
    """
    merge all dpsi for common regions (as described in big_merged_bedtool) into one dataframe. 
    """
    progress = tnrange(len(all_annotations))
    
    # do this once to easily/automatically populate the index.
    merged = transform_individual_rmats_positions(all_annotations[0], big_merged_bedtool)
    progress.update(1)
    
    # foreach subsequent file, merge (outer join to not miss any) files into merged.
    for annotation in all_annotations[1:]:
        df = transform_individual_rmats_positions(annotation, big_merged_bedtool, es, ee)
        merged = pd.merge(merged, df, how='outer', left_index=True, right_index=True)
        progress.update(1)
    return merged



rbps = []
tpm = pd.read_table(exprs)
tpm.ix[0] = ['ENSG00000117308.10','ENSG00000117308.10',1.5,'ENSG00000117308.10',1.5]
splice_files = glob.glob(os.path.join(splice_dir,'*.significant.txt'))
for splice in splice_files:
    rbps.append(os.path.basename(splice).split('-')[0])
rbps = set(rbps)
progress = tnrange(len(rbps))
for rbp in rbps:
    hepg2 = glob.glob(os.path.join(splice_dir,'{}-*-{}*significant.txt'.format(rbp,'HepG2')))
    k562 = glob.glob(os.path.join(splice_dir,'{}-*-{}*significant.txt'.format(rbp,'K562')))
    if(len(hepg2)==1) and (len(k562)==1):
        hepg2 = hepg2[0]
        k562 = k562[0]
        annotations = [k562, hepg2]
        """big_bedtool = rmats2bedtool_all(annotations, 'upstreamES', 'downstreamEE')
        x = transform_individual_rmats_positions(k562, big_bedtool,  'upstreamES', 'downstreamEE')
        y = transform_individual_rmats_positions(hepg2, big_bedtool,  'upstreamES', 'downstreamEE')"""
        big_bedtool = rmats2bedtool_all(annotations, 'exonStart_0base', 'exonEnd')
        x = transform_individual_rmats_positions(k562, big_bedtool,  'exonStart_0base', 'exonEnd')
        y = transform_individual_rmats_positions(hepg2, big_bedtool,  'exonStart_0base', 'exonEnd')
        try:
            merged = pd.merge(x, y, how='inner', left_index=True, right_index=True).reset_index()
            result = pd.merge(merged, tpm, how='inner', left_on='name', right_on='ensg')
            if result.shape[0] > 0:
                print(k562, hepg2)
        except TypeError:
            print(x.shape, y.shape), 
    if len(hepg2) > 1:
        pass
    if len(k562) > 1:
        pass
    progress.update(1)



print(tpm.shape)



tpm[tpm['ensg']=='ENSG00000057757.5']



