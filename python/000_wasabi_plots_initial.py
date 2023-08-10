from __future__ import print_function

import glob
bam_filenames = glob.glob('/home/obotvinnik/projects/singlecell_pnms/analysis/bams_from_aws/*.bam')

sample_id = 'M2_02'

bam_filename = [x for x in bam_filenames if sample_id in x][0]
bam_filename

# figwidth = 4
# figheight = len(bam_filenames) * 1
# figsize = figwidth, figheight
# print figsize

import pysam
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk')

get_ipython().run_line_magic('matplotlib', 'inline')

type(read)

pysam.__version__

from outrigger.region import Region

snap25_mxe_event = 'exon:chr20:10265372-10265420:+@exon:chr20:10273530-10273647:+@exon:chr20:10273809-10273926:+@exon:chr20:10277573-10277698:+'

snap25_exons = snap25_mxe_event.split('@')
snap25_regions = map(Region, snap25_exons)
snap25_regions

import HTSeq

bam_htseq = HTSeq.BAM_Reader(bam_filename)

get_ipython().run_line_magic('pinfo', 'bam_htseq')

get_ipython().system('samtools index $bam_filename')

# exon1 = snap25_regions[0]

# list(bam_htseq[HTSeq.GenomicInterval(snap25_regions[0].chrom, snap25_regions[0].start, 
#                                      snap25_regions[-1].stop, snap25_regions[0].strand)])

import numpy as np

HTSeq.cigar_operation_names

INSERTION_DELETIONS = ('I', 'D')


def skip_bad_cigar(read, bam_filename, bad_cigar=INSERTION_DELETIONS):
    # Skip reads with no CIGAR string
    if read.cigar is None:
        print("Skipping read with no CIGAR string: {read_name} (from {bam})".format(
                read_name=read.read.name, bam=bam_filename))
        return

    # Check if the read contains an insertion (I)
    # or deletion (D) -- if so, skip it
    for cigar_operation in read.cigar:
        cigar = cigar_operation.type
        if cigar in bad_cigar:
            print("Skipping read with CIGAR string {abbrev} (a base in the read was {full}): {read_name} (from {bam})".format(
                read_name=read.read.name, bam=bam_filename, abbrev=cigar, 
                    full=HTSeq.cigar_operation_names[cigar]))
            return
    return read



def count_region_reads(bam_htseq, bam_filename, chrom, start, stop, strand,
                       allowed_cigar=('M'), bad_cigar=('I', 'D'), offset=True):
    """Get the number of reads that matched to the reference sequence
    
    Parameters
    ----------
    bam_htseq : HTSeq.BAM_Reader
        Bam file object to get reads from
    bam_filename : str
        Name of the bam filename for logging purposes
    chrom : str
        Name of the reference chromosome
    start, stop : int
        Genome-based locations of the start and stop regions
    pad : int, optional
        Add a few nucleotides to the left and right of the array for
        visually pleasing padding. (default=10)
    allowed_cigar : tuple of str, optional
        Which CIGAR string flags are allowed. (default=('M') aka match)
    bad_cigar : tuple of str, optional
        Which CIGAR string flags are not allowed. (default=('I', 'D') aka 
        insertion and deletion)
    offset : bool, optional
        If True, offset the region counts so that the array starts at zero 
        and not the whole chromosome. Useful for plotting just the one region.
        (default=True)
    
    Returns
    -------
    counts : numpy.array
        Number of reads that matched to the genome at every location
    """

    # Add a small amount of nucleotides to left and right for making the plot look nice
#     start = start - pad
#     stop = stop + pad
    
    length = stop - start + 1 if offset else stop

    counts = np.zeros(shape=(length), dtype=int)
    
    region_reads = bam_htseq[HTSeq.GenomicInterval(chrom, start, stop, strand)]

    print('start:', start)
    
    for read in region_reads:
        read = skip_bad_cigar(read, bam_filename, bad_cigar)
        if read is None:
            continue
        for cigar_operation in read.cigar:
            # Only count where the read matched to the genome
            if cigar_operation.type not in allowed_cigar:
                continue
            match_start = cigar_operation.ref_iv.start
            match_stop = cigar_operation.ref_iv.end
            
            if offset:
                match_start = match_start - start
                match_stop = match_stop - start
                
            if match_stop < 0:
                # If the match_stop is negative, that means we have the other read of the paired end read
                # that mapped to somewhere else in the genome
                continue
            match_start = max(match_start, 0)
            match_stop = min(match_stop, length)

            counts[match_start:match_stop] += 1
    return counts


chrom = snap25_regions[0].chrom
start = snap25_regions[0].start
stop = snap25_regions[-1].stop
strand = snap25_regions[0].strand

snap25_counts = count_region_reads(bam_htseq, bam_filename, chrom, start, stop, strand)
snap25_counts

fig, ax = plt.subplots(figsize=(12, 1))

xmax = snap25_counts.shape[0]
xvalues = np.arange(0, xmax)

ax.fill_between(xvalues, snap25_counts, y2=0, linewidth=0);
ax.set(xlim=(0, xmax))

from collections import defaultdict, Counter

def pad_location(start, stop, pad):
    start = start - pad
    stop = stop + pad
    return start, stop

region_reads = bam_htseq[HTSeq.GenomicInterval(chrom, start, stop, strand)]
bad_cigar = INSERTION_DELETIONS

junctions = Counter()

# pad = 10
# start, stop = pad_location(start, stop, pad)

length = stop - start + 1

for read in region_reads:
#     read = skip_bad_cigar(read, bam_filename, bad_cigar)
    if read is None:
        continue
    for cigar_operation in read.cigar:
        # N = did not match to genome and is an insertion
        if cigar_operation.type == 'N':
            junction_start = cigar_operation.ref_iv.start - start
            junction_stop = junction_start + cigar_operation.ref_iv.length #- start
            if (junction_stop < 0) or (junction_stop > length):
                continue
            
#             print(junction_start, junction_stop, cigar_operation.ref_iv.length, 
#                   cigar_operation.ref_iv.start,
#                   cigar_operation.ref_iv.end)
            junctions[(junction_start, junction_stop)] += 1
junctions

def count_junctions(bam, chrom, start, stop, strand, bad_cigar=INSERTION_DELETIONS, offset=True):
    region_reads = bam[HTSeq.GenomicInterval(chrom, start, stop, strand)]
    junctions = Counter()

    length = stop - start + 1

    for read in region_reads:
        if read is None:
            continue
        for cigar_operation in read.cigar:
            # N = did not match to genome and is an insertion
            if cigar_operation.type == 'N':
                junction_start = cigar_operation.ref_iv.start
                
                if offset:
                    junction_start -= start
                    
                junction_stop = junction_start + cigar_operation.ref_iv.length
                
                if (junction_stop < 0) or (junction_stop > length):
                    # If any of the junctions start or end outside of the region, skip it
                    continue
                junctions[(junction_start, junction_stop)] += 1
    return junctions


def cubic_bezier(points, t):
    """
    Get points in a cubic bezier.
    """
    p0, p1, p2, p3 = points
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    return p0 * (1 - t)**3 + 3 * t * p1 * (1 - t) ** 2 +         3 * t**2 * (1 - t) * p2 + t**3 * p3

import matplotlib as mpl

mpl.artist.Path

ymin = 1

from matplotlib.artist import Path
from matplotlib.patches import PathPatch

fig, ax = plt.subplots()
xmax = snap25_counts.shape[0]
xvalues = np.arange(0, xmax)

ax.fill_between(xvalues, snap25_counts, y2=0, linewidth=0);
ax.set(xlim=(0, xmax))


for (left, right), n_junction_reads in junctions.items():
    print(left, right)
#     midpoint = (right - left)/2
    curve_height = 10 # 3 * float(ymin) / 4
    
    
    left_height = snap25_counts[left-1]
    right_height = snap25_counts[right+1]
    print('\t', left_height, right_height)
    vertices = [(left, left_height), 
                (left, left_height + curve_height), 
                (right, right_height + curve_height), 
                (right, right_height)]
    print('\t\t', vertices)
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    midpoint = cubic_bezier(vertices, 0.5)
    
    if n_junction_reads:
        plt.text(midpoint[0], midpoint[1], '{}'.format(n_junction_reads),
             fontsize=6, ha='center', va='center', backgroundcolor='w')

    path = Path(vertices, codes)
    patch = PathPatch(path, ec='black', lw=np.log10(n_junction_reads + 1), fc='none')
    ax.add_patch(patch)
    
#     xs, ys = zip(*vertices)
#     ax.plot(xs, ys, 'x--', lw=2, color='black', ms=10)

#     for i, (x, y) in enumerate(zip(xs, ys)):
#         ax.text(x, y, 'P{}'.format(i))
#     ax.text(0.15, 1.05, 'P1')
#     ax.text(1.05, 0.85, 'P2')
#     ax.text(0.85, -0.05, 'P3')

def junctionplot(left, right, n_junction_reads, read_counts, curve_height_multiplier=0.1, #curve_height=10, 
                 color=None,
                 text_kws=dict(fontsize=6, horizontalalignment='center', verticalalignment='center', backgroundcolor='w'),
                 linewidth_function=lambda x: np.log10(x + 1),
                 patch_kws=dict(facecolor='none',)):
    """
    
    Uses the y-axis limits to determine the curve height so make sure to use this function AFTER 
    
    Parameters
    ----------
    
    patch_kws : dict
        To change the color of the line, use the "edgecolor" keyword, e.g. dict(edgecolor="black")
    """
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    curve_height = yrange * curve_height_multiplier
    
    left_height = read_counts[left-1]
    right_height = read_counts[right+1]
#     print('\t', left_height, right_height)
    vertices = [(left, left_height), 
                (left, left_height + curve_height), 
                (right, right_height + curve_height), 
                (right, right_height)]
#     print('\t\t', vertices)
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    midpoint = cubic_bezier(vertices, 0.5)
    
    if n_junction_reads:
        plt.text(midpoint[0], midpoint[1], '{}'.format(n_junction_reads), **text_kws)

    path = Path(vertices, codes)
    
    patch_kws['linewidth'] = linewidth_function(n_junction_reads)
    if color is not None:
        patch_kws['edgecolor'] = color
    patch = PathPatch(path, **patch_kws)
    return ax.add_patch(patch)

color = 'steelblue'

snap25_counts_log = np.log10(snap25_counts+1)
snap25_counts_log

def coverageplot(counts, color, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    xmax = counts.shape[0]
    xvalues = np.arange(0, xmax)
#     ax.set(xlim=(0, xmax))
    
    return ax.fill_between(xvalues, counts, y2=0, linewidth=0, color=color, **kwargs)

fig, ax = plt.subplots()
xmax = snap25_counts.shape[0]
xvalues = np.arange(0, xmax)

ax.fill_between(xvalues, snap25_counts_log, y2=0, linewidth=0, color=color);
ax.set(xlim=(0, xmax))

ymin, ymax = ax.get_ylim()
yrange = ymax - ymin

for (left, right), n_junction_reads in junctions.items():
    junctionplot(left, right, n_junction_reads, snap25_counts_log, color=color, curve_height_multiplier=0.2)
yticks = map(int, ax.get_yticks())
yticklabels = ('$10^{}$'.format(y) for y in yticks)
ax.set(yticks=yticks, yticklabels=yticklabels, ylabel='reads')

# class WasabiPlotter(object):
    
#     def __init__(bam_filename, chrom, start, stop, log_base, ax):
#         self.

def wasabiplot(bam_filename, chrom, start, stop, strand, log_base=10, ax=None):
    bam = HTSeq.BAM_Reader(bam_filename)
    counts = count_region_reads(bam, bam_filename, chrom, start, stop, strand)
    counts = np.log(counts)/np.log(log_base)
    
    if ax is None:
        ax = plt.gca()
        
    coverageplot(counts, color, ax)

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    junctions = count_junctions(bam, chrom, start, stop, strand)
    
    for (left, right), n_junction_reads in junctions.items():
        junctionplot(left, right, n_junction_reads, counts, color=color, curve_height_multiplier=0.2)
    yticks = map(int, ax.get_yticks())
    yticklabels = ('$10^{}$'.format(y) for y in yticks)
    ax.set(yticks=yticks, yticklabels=yticklabels, ylabel='reads')

bam_filename2 = [x for x in bam_filenames if 'M2_03' in x][0]

get_ipython().system(' ls $bam_filename2*')

wasabiplot(bam_filename2, chrom, start, stop, strand)

cigar_operation.ref_iv.start

cigar_operation = read.cigar[1]
cigar_operation

cigar_operation.ref_iv.start

cigar_operation.ref_iv.length

cigar_operation.ref_iv.end



stop

cigar_operation.ref_iv.end_d

cigar_operation.ref_iv.end

cigar_operation.ref_iv.start

cigar_operation.check()

read.iv.start

# subset_reads = bam.fetch(reference=exon1.chrom, start=snap25_regions[0].start, end=snap25_regions[-1].stop)
# subset_reads



