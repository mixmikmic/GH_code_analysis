import pandas as pd
import gffutils
import os
from collections import defaultdict

annotation_file = '/projects/ps-yeolab/genomes/mm10/gencode/gencode.vM15.annotation.gtf'
# annotation_file = '/home/bay001/projects/codebase/annotator/test/c_elegans.PRJNA13758.WS257.canonical_geneset.chrIII.25000.gtf'
db_file = '/projects/ps-yeolab/genomes/mm10/gencode/gencode.vM15.annotation.gtf.db'

def build_db(annotation_file, db_file, force=True, disable_infer_genes=True, disable_infer_transcripts=True):
    db = gffutils.create_db(
        annotation_file, dbfn=db_file, force=force, # change to True if we need to create a new db
        keep_order=True, merge_strategy='merge', sort_attribute_values=True,
        disable_infer_genes=disable_infer_genes,
        disable_infer_transcripts=disable_infer_transcripts
    )
build_db(annotation_file, db_file, disable_infer_genes=False, disable_infer_transcripts=False)

# db_file = '/projects/ps-yeolab/genomes/mm10/gencode/gencode.vM3.annotation.gtf.db'
# db_file = '/projects/ps-yeolab/genomes/hg19/gencode_v19/gencode.v19.annotation.gtf.db'
db_file = '/projects/ps-yeolab3/bay001/annotations/gencode.vM10.annotation.v3.db'
DATABASE = gffutils.FeatureDB(db_file)

def gene_id_to_name(db):
    '''
    Returns a dictionary containing a gene_id:name translation
    Note: may be different if the 'gene_id' or 'gene_name' 
    keys are not in the source GTF file
    (taken from gscripts.region_helpers)
    '''
    genes = db.features_of_type('gene')
    gene_name_dict = {}
    for gene in genes:
        gene_id = gene.attributes['gene_id'][0] if type(gene.attributes['gene_id']) == list else gene.attributes['gene_id']
        try:
            gene_name_dict[gene_id] = gene.attributes['gene_name'][0]
        except KeyError:
            print(gene.attributes.keys())
            print("Warning. Key not found for {}".format(gene))
            return 1
    return gene_name_dict

gene_id_to_name_dictionary = gene_id_to_name(DATABASE)
# gene_id_to_name_dictionary['ENSG00000100320.18']
gene_id_to_name_dictionary['ENSMUSG00000092210.1']

def gene_id_to_protein_coding(db):
    """
    returns whether or not a gene is protein coding or not.
    """
    genes = db.features_of_type('gene')
    gene_name_dict = {}
    for gene in genes:
        gene_id = gene.attributes['gene_id'][0] if type(gene.attributes['gene_id']) == list else gene.attributes['gene_id']
        try:
            gene_name_dict[gene_id] = gene.attributes['gene_type'][0]
        except KeyError:
            print(gene.attributes.keys())
            print("Warning. Key not found for {}".format(gene))
            return 1
    return gene_name_dict

gene_id_to_pc = gene_id_to_protein_coding(DATABASE)
gene_id_to_pc['ENSMUSG00000092210.1']

def gene_name_to_id(db):
    '''
    given a gene name, returns a list of associated Gene IDs (one-to-many)
    '''
    genes = db.features_of_type('gene')
    gene_name_dict = defaultdict(list)
    for gene in genes:
        try:
            gene_name_dict[gene.attributes['gene_name'][0]].append(gene.attributes['gene_id'][0])
        except KeyError as e:
            print("Warning. Key not found for {}".format(gene))
            return 1
    return gene_name_dict

gene_name_to_id_dictionary = gene_name_to_id(DATABASE)
gene_name_to_id_dictionary['RBFOX2']

def gene_name_to_transcript(db):
    '''
    given a gene name, returns a list of associated transcript IDs (one-to-many)
    '''
    genes = db.features_of_type('transcript')
    gene_name_dict = defaultdict(list)
    for gene in genes:
        try:
            gene_name_dict[gene.attributes['gene_name'][0]].append(gene.attributes['transcript_id'][0])
        except KeyError as e:
            print("Warning. Key not found for {}".format(gene))
            return 1
    return gene_name_dict

gene_name_to_id_dictionary = gene_name_to_transcript(DATABASE)
gene_name_to_id_dictionary['RBFOX2']

def id_to_exons(db, identifier):
    '''
    takes the gene or transcript id and returns exon positions
    '''
    exons = []
    for i in db.children(identifier, featuretype='exon', order_by='start'):
        exons.append(i)
    return exons

id_to_exons(DATABASE,'ENST00000473487.2')

def position_to_features(db, chrom, start, end, strand='', completely_within=True):
    '''
    takes a coordinate and returns all the features overlapping 
    (either completely contained or partially overlapping the region).
    '''
    if strand == '+' or strand == '-':
        return list(
            db.region(
                region=(chrom, start, end), strand=strand, completely_within=completely_within
            )
        )
    else:
        return list(
            db.region(
                region=(chrom, start, end), completely_within=completely_within
            )
        )
# get all features corresponding to the genomic coordinates (True if feature must be entirely contained within region)
features = position_to_features(DATABASE,'chr19', 1000000, 1000100, completely_within=True)
# print all gene names associated with these features
# print([f.attributes['gene_name'] for f in features])

from collections import defaultdict


def hash_features(db):
    '''
    hashes features by position.
    '''
    genes = defaultdict(list)
    for element in db.region(seqid=chrom):
        start = int(element.start / 1000000)
        end = int(element.end / 1000000)
        genes[chrom, start, end].append(element)
    return genes
# get all features corresponding to the genomic coordinates (True if feature must be entirely contained within region)
genes = chrom_to_features(DATABASE,'chr19')
# print all gene names associated with these features
# print([f.attributes['gene_name'] for f in features])

start = 1000400
end = 1000440

overlapped = []

start_key = int(start / 1000000)
end_key = int(end / 1000000)

for gene in genes[chrom, start_key, end_key]:
    if gene.start > start and gene.start < end:
        overlapped.append(gene)
    elif gene.end > start and gene.end < end:
        overlapped.append(gene)
        
overlapped

ret = DATABASE.execute("SELECT seqid FROM features").fetchall()
all_chromosomes = [r['seqid'] for r in ret]

from tqdm import tnrange, tqdm_notebook

genes = DATABASE.features_of_type('gene')
progress = tnrange(48440)

ct = 0
newgenes = []
for gene in genes:
    if ct > 10000:
        break
    gene.attributes['transcript_id'] = gene.attributes['gene_id']
    newgenes.append(gene)
    progress.update(1)
    ct+=1

DATABASE.update((n for n in newgenes), )

# Feature objects embed all information as a dictionary
# See: http://pythonhosted.org/gffutils/attributes.html

DEFAULT_FEATURE_TYPE_PRIORITY = [
    'UTR','gene','transcript','exon','start_codon','stop_codon','Selenocysteine', 'CDS'
]

DEFAULT_TRANSCRIPT_TYPE_PRIORITY = [
    'retained_intron', 'protein_coding','pseudogene','rRNA', 'processed_transcript', 'antisense'
]

priority = DEFAULT_TRANSCRIPT_TYPE_PRIORITY
"""
for f in features:
    pass
    print(
        '{}, {}, {}, {}'.format(
            f.attributes['gene_name'], # list of associated gene names
            f.start, # start of feature
            f.end, # end of feature
            priority.index(f.attributes['transcript_type'][0])
        ) # type of feature
    )"""

f_priority = DEFAULT_FEATURE_TYPE_PRIORITY
t_priority = DEFAULT_TRANSCRIPT_TYPE_PRIORITY

features.sort(
    key=lambda x: t_priority.index(
        x.attributes['gene_type'][0]
    ), reverse=False
) # sort gene type
first_filter = [
    f for f in features if features[0].attributes['transcript_type'] == f.attributes['transcript_type']
]
first_filter.sort(
    key=lambda x: f_priority.index(
        x.featuretype
    ), reverse=False
)
second_filter = [
    f for f in first_filter if first_filter[0].featuretype == f.featuretype
]
# [f.attributes['transcript_type'] for f in features]
second_filter

get_highest_priority_annotation()

df = pd.read_table('/projects/ps-yeolab3/bay001/annotations/small_bed.bed3', names=['chrom','start','end'], index_col=0)
df

features = {}
for ix, row in df.iterrows():
    features[ix] = position_to_features(DATABASE, row['chrom'], row['start'], row['end'], True)

for name, region_list in features.iteritems():
    for region in region_list:
        print(name, region.featuretype, region.attributes['transcript_type'][0], region.attributes['gene_name'][0])

import pybedtools

interval = pybedtools.create_interval_from_list(['chr1','13000000','13003000','some_interval','0','-'])

def bedtool_to_features(db, interval, completely_within):
    """
    
    takes a coordinate and returns all the features overlapping 
    (either completely contained or partially overlapping the region).
    
    Parameters
    ----------
    db : sqlite3 database
    interval : pybedtools.Interval
        interval object
    completely_within : bool
        True if the features returned must be completely contained
        within the region. False if the features need only to be
        partially overlapping the region.
        
    Returns
    -------
    region_list: list
        list of Features corresponding to overlapping/contained
        features intersecting a region.
    """
    return position_to_feature(
        db,
        interval.chrom,
        interval.start,
        interval.end,
        interval.strand,
        completely_within
    )

bedtool_to_features(DATABASE, interval, True)

gtf_file = '/projects/ps-yeolab/genomes/hg19/gencode_v19/gencode.v19.annotation.gtf'
# gtf_file = '/projects/ps-yeolab3/bay001/annotations/c_elegans.PRJNA13758.WS257.canonical_geneset.gtf'
GTF_NAMES = ['chrom','source','feature_type','start','end','.','strand','.','attributes']


def get_feature_type_set(gtf_file):
    """
    from a GTF file, extract the set of feature_types
    (feature_types is the third column, normally)
    This might be useful for figuring out the priority for annotation.
    
    Parameters
    ----------
    gtf_file

    Returns
    -------

    """
    gtf_df = pd.read_table(
        gtf_file,
        names=GTF_NAMES,
        comment='#'
    )
    return set(gtf_df['feature_type'])


def get_attribute_type_set(gtf_file, attribute_type):
    """
    from a GTF file, extract the set of attribute_types
    (attribute_types is one of those fields contained within the 9th column)
    This might be useful for figuring out the priority for annotation.
    
    Parameters
    ----------
    gtf_file : basestring
    attribute_type : basestring

    Returns
    -------

    """

    gtf_df = pd.read_table(
        gtf_file,
        names=GTF_NAMES,
        comment='#'
    )
    regex_filter = '{} \"([\w\s\d -]+)\"'.format(attribute_type)
    return set(gtf_df['attributes'].str.extract(regex_filter, expand=False))

# in C elegans GFF
get_feature_type_set(gtf_file)

# in C elegans GFF
get_attribute_type_set(gtf_file, 'biotype')

# in Human GENCODE
get_feature_type_set(gtf_file)

# in Human GENCODE
get_attribute_type_set(gtf_file, 'transcript_type')

df = pd.read_table(gtf_file, names=GTF_NAMES, comment='#')
df.head()

df = df[(df['chrom'] == 'chr19') & (df['end'] <= 550000)]
df.tail()

df.to_csv(
    '/projects/ps-yeolab3/cellrangerdatasets/hg19chr19kbp550_CELLRANGER_REFERENCE/gencode.v19.chr19.550000.gtf',
    sep='\t', index=False, header=False
)



