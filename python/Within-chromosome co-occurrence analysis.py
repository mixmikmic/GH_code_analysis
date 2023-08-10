import sys
sys.path.append("../lib")

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

import numpy
import scipy.stats

import nbsupport.io
import nbsupport.plots
import nbsupport.tcga

import discover

dataFile = "../data/tcga/tcga-pancan12.h5"

events = nbsupport.io.load_discover_matrix(dataFile, "/models/combined")

result_cooc = discover.pairwise_discover_test(events, alternative="greater")

segments = {}

for event_type in ["amp", "del"]:
    peaks = nbsupport.tcga.read_gistic_output("../data/tcga/%s_genes.conf_95.pancan12.txt" % event_type)
    segments.update({ gene.strip("[]") + "_" + {"amp": "gain", "del": "loss"}[event_type]: seg for seg in peaks for gene in peaks[seg] })

seg = numpy.array([segments.get(gene, "NA") for gene in events.rownames])

chrom_arm = numpy.array([x if x == "NA" else x[:max(x.find("p"), x.find("q"))+1] for x in seg])

from numpy import newaxis

same_segment = seg[:, newaxis] == numpy.char.replace(seg[newaxis], "NA", "NA2")
same_arm = ~same_segment & (chrom_arm[:, newaxis] == numpy.char.replace(chrom_arm[newaxis], "NA", "NA2"))

chrom = numpy.char.replace(numpy.char.replace(chrom_arm, "p", ""), "q", "")
same_chrom = ~same_segment & ~same_arm & (chrom[:, newaxis] == numpy.char.replace(chrom[newaxis], "NA", "NA2"))

rest = ~same_segment & ~same_arm & ~same_chrom & (seg != "NA")[:, newaxis] & (seg != "NA")[newaxis]

data = [numpy.ma.masked_invalid(result_cooc.pvalues.values[same_segment]).compressed(),
        numpy.ma.masked_invalid(result_cooc.pvalues.values[same_arm]).compressed(),
        numpy.ma.masked_invalid(result_cooc.pvalues.values[same_chrom]).compressed(),
        numpy.ma.masked_invalid(result_cooc.pvalues.values[rest]).compressed()]

with plt.rc_context(rc={"font.size": 12,
                        "legend.fontsize": 12,
                        "axes.linewidth": 1.5,
                        
                        'mathtext.fontset': 'custom',
                        'mathtext.rm': 'Arial',
                        'mathtext.it': 'Arial:italic',
                        'mathtext.bf': 'Arial:bold',
                        
                        "font.family": "arial"}):
    bp = plt.boxplot(data, widths=0.25)
    
    ax = plt.gca()
    
    plt.setp(bp["boxes"], color="#31a354", lw=0)
    plt.setp(bp["medians"], color="white")
    plt.setp(bp["medians"][0], color="#0072b2")
    plt.setp(bp["whiskers"], color="#4393c3", lw=2)
    plt.setp(bp["caps"], color="black")
    plt.setp(bp["fliers"], color="red")
    
    for box in bp["boxes"]:
        coords = zip(box.get_xdata(), box.get_ydata())
        ax.add_patch(plt.Polygon(coords, fc="#0072b2", lw=0))
    
    for i in xrange(len(data) - 1):
        y = 1.03 if i % 2 == 0 else 1.2
        ax.annotate("", xy=(i + 1, y), xycoords="data", xytext=(i + 2, y), textcoords="data",
                    arrowprops={"arrowstyle": "-",
                                "ec": "#aaaaaa",
                                "connectionstyle": "bar,fraction=0.2"})
        p = scipy.stats.mannwhitneyu(data[i], data[i + 1]).pvalue
        ptext = ("$%.2g}$" % p).replace("e", "\\times 10^{")
        ax.text(i + 1.5, y + 0.17, ptext, ha="center", va="center")

    plt.ylim(-0.01, 1.5)
    plt.xticks([1, 2, 3, 4], ["Same segment", "Same arm", "Same chromosome", "Rest"])
    
    ax.axes.spines["left"].set_bounds(0, 1)
    ax.yaxis.set_ticks(numpy.linspace(0, 1, 6))
    ax.axes.spines['top'].set_visible(False)
    ax.axes.spines['right'].set_visible(False)
    ax.axes.spines['bottom'].set_visible(False)
    ax.axes.yaxis.set_ticks_position('left')
    ax.axes.xaxis.set_ticks_position('none')
    ax.axes.tick_params(direction="out", which="both")
    ax.spines['bottom'].set_position(('outward', 10))
    
    yPos = ax.transAxes.inverted().transform(ax.transData.transform_point((0, 0.5)))[1]
    plt.ylabel("P value", y=yPos)

print "Pairs in same segment:"
print "  tested:     ", numpy.isfinite(result_cooc.pvalues.values[same_segment]).sum()
print "  significant:", (result_cooc.qvalues.values[same_segment] * result_cooc.pi0 < 0.01).sum()

print "Pairs on same chromosome arm:"
print "  tested:     ", numpy.isfinite(result_cooc.pvalues.values[same_arm]).sum()
print "  significant:", (result_cooc.qvalues.values[same_arm] * result_cooc.pi0 < 0.01).sum()

print "Pairs on same chromosome:"
print "  tested:     ", numpy.isfinite(result_cooc.pvalues.values[same_chrom]).sum()
print "  significant:", (result_cooc.qvalues.values[same_chrom] * result_cooc.pi0 < 0.01).sum()

