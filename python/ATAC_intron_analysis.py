get_ipython().magic('matplotlib inline')
from collections import defaultdict
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pybedtools
import glob
import gffutils
from qtools import Submitter
from tqdm import tnrange, tqdm_notebook

gtf_db = '/projects/ps-yeolab/genomes/hg19/gencode_v19/gencode.v19.annotation.gtf.db'

MAXVAL=5000000000
MINVAL=-1
starts = defaultdict(lambda: MINVAL)
ends = defaultdict(lambda: MAXVAL)

db = gffutils.FeatureDB(gtf_db)
for exon in db.features_of_type('exon'):
    ends[exon.start] = min(ends[exon.start], exon.end)
    starts[exon.end] = max(starts[exon.end], exon.start)

starts[54405770]

sjout_tab_names = ['chrom','start1based','stop1based','strand','intron_motif','annotation','uniq','multimap','maxoverhang']
sjout_dir = '/home/bay001/projects/encode/analysis/atac_intron_analysis/SJout/'

# sjout_tab = os.path.join(sjout_dir, 'RBFOX2-204-CLIP_S1_R1.A01_204_01_RBFOX2.adapterTrim.round2.rmRep.bamSJ.out.tab')
# sjout_df = pd.read_table(sjout_tab, names=sjout_tab_names)
# sjout_df.head()
sjout_files = glob.glob(os.path.join(sjout_dir, '*.tab'))
sjout_files[:5]

def split_filename_and_get_uid_collection(files):
    """
    Splits the filename based on some kind of structure 
    and returns the uid as a dict.
    
    Processes unassigned/input separately.
    
    uid_collection = {uid:[rep1, rep2], uid:[rep1, rep2]}
    
    """
    uid_collection = defaultdict(list)
    for f in files:
        base = os.path.basename(f)
        prefix = base.split('.')
        prefix1 = prefix[0]
        prefix2 = prefix[1]
        if 'unassigned' in prefix2:
            uid = prefix1.split('_')[0]
        else:
            uid = '_'.join([prefix2.split('_')[1], prefix2.split('_')[2]])
        uid_collection[uid].append(f)
    return uid_collection

def combine_sjout_dataframes_and_filter(f1, f2, f):
    """
    Combines the two sjout dataframes and returns a
    singular filtered dataframe containing only
    annotated introns (1) and ATAC motifs (5).
    """
    df1 = pd.read_table(f1, names=sjout_tab_names)
    df2 = pd.read_table(f2, names=sjout_tab_names)
    df = pd.concat([df1, df2])
    df.drop_duplicates(inplace=True, subset=['chrom','start1based','stop1based','strand'])
    atac = df[(df['intron_motif']==f) & (df['annotation']==1)]
    return atac

### Get all UIDs ###

uids = split_filename_and_get_uid_collection(sjout_files)

MIN_NUMBER_EVENTS_TO_PRINT = 50
# print the number of atac introns found 
progress = tnrange(len(uids.keys()))
pairs=0  # number of pairs found per replicate

non_canonical_list = defaultdict()
_list = defaultdict(dict)

for uid in uids.keys():
    if len(uids[uid]) == 2: # found at least 2 barcode-associated SJout files...
        pairs+=1
        
        non_canonical = combine_sjout_dataframes_and_filter(uids[uid][0], uids[uid][1], 0)
        gtag = combine_sjout_dataframes_and_filter(uids[uid][0], uids[uid][1], 1)
        ctac = combine_sjout_dataframes_and_filter(uids[uid][0], uids[uid][1], 2)
        gcag = combine_sjout_dataframes_and_filter(uids[uid][0], uids[uid][1], 3)
        ctgc = combine_sjout_dataframes_and_filter(uids[uid][0], uids[uid][1], 4)
        atac = combine_sjout_dataframes_and_filter(uids[uid][0], uids[uid][1], 5)
        gtat = combine_sjout_dataframes_and_filter(uids[uid][0], uids[uid][1], 6)
        _list[uid]['non_canonical'] = non_canonical
        _list[uid]['gtag'] = gtag
        _list[uid]['ctac'] = ctac
        _list[uid]['gcag'] = gcag
        _list[uid]['ctgc'] = ctgc
        _list[uid]['atac'] = atac
        _list[uid]['gtat'] = gtat
             
    progress.update(1)
print('% of total with pairs: {}'.format(float(pairs/len(uids.keys()))))

# kinda weird, the sjout files contain introns that are annotated, yet not found in gencode annotations.
# 629_CLIP_S21_L005_R1_001.A03_629_02_UCHL5.adapterTrim.round2.rmRep.bamSJ.out.tab
test = _list['629_02']['ctac'][['chrom','start1based','stop1based', 'strand','annotation']]
test[test['start1based']==54405771]

exons_gtf = '/projects/ps-yeolab/genomes/hg19/gencode_v19/gencode.v19.annotation.exon.gtf'
    
def get_flanking_exons(starts, ends, intron):
    """
    takes intronand finds the lower/upper exons using starts/ends dicts
    """
    # print(intron, intron.start, str(starts[intron.start]), intron.stop, str(ends[intron.stop]))
    try:
        lower_exon = pybedtools.create_interval_from_list([
            intron.chrom,
            str(starts[intron.start-1] - 1), # positions are 1-based, so must turn to 0-based.
            str(intron.start - 1),
            'name',
            '0',
            intron.strand
        ])
    except Exception as e: # handle errors in not being able to find bordering exons
        # print(e)
        # print(intron)
        lower_exon = pybedtools.create_interval_from_list([
            intron.chrom,
            str(intron.start-52),
            str(intron.start - 1),
            'name',
            '0',
            intron.strand
        ])
    try:
        upper_exon = pybedtools.create_interval_from_list([
            intron.chrom,
            str(intron.end),
            str(ends[intron.end+1]),
            'name',
            '0',
            intron.strand
        ])
    except Exception as e: # handle errors in not being able to find bordering exons
        upper_exon = pybedtools.create_interval_from_list([
            intron.chrom,
            str(intron.end),
            str(intron.end+51),
            'name',
            '0',
            intron.strand
        ])
    intron.start = intron.start - 1  # turn intron (1-based) to 0-based
    return lower_exon, intron, upper_exon

def get_strand(i):
    """
    parses the STAR-coded strand info to a string representation.
    """
    if i == 0:
        return 'u'
    elif i == 1:
        return '+'
    elif i == 2:
        return '-'
    else:
        return 'x'

def add_interval(atac):
    """
    appends bedtools.interval to each intron.
    """
    atac['interval'] = atac.apply(lambda x: pybedtools.create_interval_from_list([
        x['chrom'], x['start1based'], x['stop1based'], 'name', '0', get_strand(x['strand'])
    ]), axis=1)
    return atac

def to_twobed(lower, upper):
    """
    Just concatenates the two BedTools into a single line.
    """
    return '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        lower.chrom, lower.start, lower.end, lower.name, lower.score, lower.strand,
        upper.chrom, upper.start, upper.end, upper.name, upper.score, upper.strand
    )

# http://labshare.cshl.edu/shares/gingeraslab/www-data/dobin/STAR/STAR.posix/doc/STARmanual.pdf
out_dir = '/projects/ps-yeolab3/bay001/maps/current_annotations/atac_introns/'
progress = tnrange(len(dx_list.keys()))

for uid in dx_list.keys():
    for motif in ['ctac', 'gcag', 'atac', 'gtat', 'ctgc', 'gtag']:
        df = add_interval(_list[uid][motif])
        o = open(os.path.join(out_dir, '{}.{}.txt'.format(uid, motif)), 'w')
    
        for interval in df['interval']:
            lower, mid, upper = get_flanking_exons(starts, ends, interval)
            o.write(to_twobed(lower, upper))

        o.close()
    progress.update(1)

annotation_dir = '/projects/ps-yeolab3/bay001/maps/current_annotations/atac_introns/'
out_dir = '/projects/ps-yeolab3/bay001/maps/current/atac_introns/'
manifest = '/home/elvannostrand/data/clip/CLIPseq_analysis/ENCODE_FINALforpapers_20170325/ALLDATASETS_submittedonly.txt'
manifest_df = pd.read_table(manifest)
manifest_df.head(2)

prog = '/home/bay001/projects/codebase/rbp-maps/maps/plot_density.py'
chrom_sizes = '/projects/ps-yeolab/genomes/hg19/hg19.chrom.sizes'
cmds = []

def find_annotations_from_uid(annotation_dir, uid):
    
    found_files = glob.glob(os.path.join(annotation_dir,'{}.*.txt'.format(uid)))
    # found_major_files = glob.glob(os.path.join(annotation_dir,'{}.major.txt'.format(uid)))
    # found_minor_files = glob.glob(os.path.join(annotation_dir,'{}.minor.txt'.format(uid)))
    if len(found_files) > 1:
        return found_files
    else:
        return None
    
for col, row in manifest_df.iterrows():
    inp_file = row['INPUT']
    uid = row['uID']
    for rep in ['{}_01'.format(uid),'{}_02'.format(uid)]:
        if rep.endswith('_01'):
            rep_file = row['CLIP_rep1']
        elif rep.endswith('_02'):
            rep_file = row['CLIP_rep2']
        else:
            print(row)
        
        out_file = os.path.join(out_dir, os.path.basename(rep_file).replace('.bam','.png'))
        annotations = find_annotations_from_uid(annotation_dir, rep)
        if annotations is not None:
            cmd = 'python {} '.format(prog)
            cmd = cmd + '--event atac '
            cmd = cmd + '--annotations '
            for annotation in sorted(annotations):
                cmd = cmd + annotation + ' '
            cmd = cmd + '--annotation_type '
            for annotation in annotations:
                cmd = cmd + 'twobed' + ' '
            cmd = cmd + '--chrom_sizes {} '.format(chrom_sizes)
            cmd = cmd + '--confidence 1 '
            cmd = cmd + '--normalization_level 1 '
            cmd = cmd + '--exon_offset 50 '
            cmd = cmd + '--intron_offset 300 '
            cmd = cmd + '--ipbam {} '.format(rep_file)
            cmd = cmd + '--inputbam {} '.format(inp_file)
            cmd = cmd + '--output {} '.format(out_file)
            cmds.append(cmd)

job_name = 'atac_introns'
sh = '/home/bay001/projects/encode/analysis/atac_intron_analysis/bash_scripts/run_atac_maps.sh'
Submitter(cmds, job_name=job_name, sh=sh, queue='home-yeo', array=True, submit=True)

cmds

df = pd.read_table(
    '/home/bay001/projects/encode/analysis/atac_intron_analysis/SJout/262_CLIP_GAGATTCC-TATAGCCT_L006_R1.A01_262_01_SLBP.adapterTrim.round2.rmRep.bamSJ.out.tab',
    names=['chrom','start','stop','strand','motif','annotated','uniq','mm','max']
)

df[df['motif']==5]



