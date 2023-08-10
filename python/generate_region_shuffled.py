import sys
sys.path.insert(0, '/home/bay001/projects/codebase/clip_analysis/clip_analysis/src/')
import pybedtools

import CLIP_analysis
from tqdm import tnrange, tqdm_notebook

data_dir = '/home/gpratt/clipper/clipper/data/'
assigned_dir = '/home/bay001/projects/encode/analysis/conservation_analysis/assigned_random_regions'

def make_clipper_ish(interval):
    interval.name = interval[7]
    interval[6] = interval.start
    interval[7] = interval.stop
    return interval

import glob
import os

wd = '/home/bay001/projects/encode/analysis/conservation_analysis/idr_peaks/'
all_beds = glob.glob(os.path.join(wd, "*.annotated"))
all_beds[:2]

progress = tnrange(len(all_beds))
for bed in all_beds:
    bed_prefix = os.path.basename(bed)
    clipper_bedtool = []
    bedtool = pybedtools.BedTool(bed)
    for interval in bedtool:
        clipper_bedtool.append(make_clipper_ish(interval))
    clipper_bedtool = pybedtools.BedTool(clipper_bedtool)
    bed_dicts = CLIP_analysis.assign_to_regions(
        clipper_bedtool, bed_prefix, data_dir=data_dir, assigned_dir=assigned_dir, nrand=1
    )
    progress.update(1)

len(clipper_bedtool)

bed = '/home/bay001/projects/codebase/annotator/testdata/539.01v02.IDR.out.0102merged.25.bed.annotated'

clipper_bedtool = []

bedtool = pybedtools.BedTool(bed)
for interval in bedtool:
    clipper_bedtool.append(make_clipper_ish(interval))

clipper_bedtool = pybedtools.BedTool(clipper_bedtool)
bed_dicts = CLIP_analysis.assign_to_regions(
    clipper_bedtool, 'test_cluster', data_dir=data_dir, assigned_dir=assigned_dir, nrand=1
)

