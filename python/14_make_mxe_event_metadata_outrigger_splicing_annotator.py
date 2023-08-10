import pandas as pd
from outrigger.region import Region
from outrigger.io.gtf import SplicingAnnotator

import gffutils

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import outrigger

outrigger.__file__

mxe_events = pd.read_csv('/home/obotvinnik/projects/singlecell_pnms/analysis/outrigger_v2/index/mxe/events.csv', index_col=0)
print(mxe_events.shape)
mxe_events.head(2)

# mxe_events_not_duplicated = mxe_events.drop_duplicates()
# print(mxe_events_not_duplicated.shape)
# mxe_events_not_duplicated.head(2)

get_ipython().run_cell_magic('time', '', "mxe_exon_bed_template = '/home/obotvinnik/projects/singlecell_pnms/analysis/outrigger_v2/index/mxe/exon{}.bed'\n\nexons = {}\n\nfor i in range(1, 5):\n    exon = 'exon{}'.format(i)\n    bed = mxe_exon_bed_template.format(i)\n    exon_df = pd.read_table(bed, names=['chrom', 'start', 'stop', 'name', 'score', 'strand'])\n    exon_regions = exon_df.apply(lambda row: Region('exon:{chrom}:{start}-{stop}:{strand}'.format(\n                chrom=row.chrom, start=row.start+1, stop=row.stop, strand=row.strand)), axis=1)\n    exons[exon] = exon_regions\nexon_regions_df = pd.DataFrame(exons)\nprint(exon_regions_df.shape)\nexon_regions_df.head()")

exon_str = exon_regions_df.applymap(lambda x: x.name)
print(exon_str.shape)
exon_str.head()

exon_str_not_duplicated = exon_str.drop_duplicates()
print(exon_str_not_duplicated.shape)
exon_str_not_duplicated.head()

exon_str.index = mxe_events.index
print(exon_str.shape)
exon_str.head(2)


outrigger_folder = '/projects/ps-yeolab/obotvinnik/singlecell_pnms/outrigger_v2'

db_filename = '{}/index/gtf/gencode.v19.annotation.gtf.db'.format(outrigger_folder)

db = gffutils.FeatureDB(db_filename)

splicing_annotator = SplicingAnnotator(db, events=exon_str, splice_type='mxe')

range(3)

3+3

attributes = splicing_annotator.attributes()

list(db.region('chr17:75211927-75212017', featuretype='exon'))

list(db.region('chr17:75211927-75212017', featuretype='gene'))



