cd /home/obotvinnik/projects/singlecell_pnms/analysis/htseq_memmap

ls -lha

get_ipython().system(' touch *')

ls -lha

get_ipython().system('head hg19_phastcons_placental_mammal_htseq.pickle')

import cPickle as pickle
memmap_dir = '/home/obotvinnik/projects/singlecell_pnms/analysis/htseq_memmap'

filename = '{}/hg19_phastcons_placental_mammal_htseq.pickle'.format(memmap_dir)
with open(filename) as f:
    conservation = pickle.load(f)

interval = 'chr17:41,198,123-41,198,198'

import HTSeq

chromvector = conservation[HTSeq.GenomicInterval('chr17', 41198123, 41198198)]

import numpy as np

np.array([x for x in chromvector.values()])

interval= 'chr10:79,797,655-79,797,710'

chromvector = conservation[HTSeq.GenomicInterval('chr10', 79797655, 79797710)]

np.array([x for x in chromvector.values()])

import pandas as pd
chromsizes = pd.read_table('/projects/ps-yeolab/genomes/hg19/hg19.chrom.sizes', header=None, index_col=0, squeeze=True)

# Remove all haplotype chromosomes
chromsizes = chromsizes[chromsizes.index.map(lambda x: '_' not in x)]
chromsizes = chromsizes.to_dict()
chromsizes

get_ipython().run_cell_magic('time', '', "import HTSeq\n\nwiggle_filename = '/projects/ps-yeolab/genomes/hg19/hg19_phastcons_placental_mammal_space_separated.wig'\nwig = HTSeq.WiggleReader(wiggle_filename)\n\nconservation = HTSeq.GenomicArray(chromsizes, stranded=False, typecode='d', storage='memmap', \n                                  memmap_dir='/home/obotvinnik/projects/singlecell_pnms/analysis/htseq_memmap')\nfor location, score in wig:\n    conservation[location] += score")

get_ipython().run_cell_magic('time', '', "\nimport cPickle as pickle\nmemmap_dir = '/home/obotvinnik/projects/singlecell_pnms/analysis/htseq_memmap'\nwith open('{}/hg19_phastcons_placental_mammal_htseq.pickle'.format(memmap_dir), 'wb') as f:\n    pickle.dump(conservation, f)")



