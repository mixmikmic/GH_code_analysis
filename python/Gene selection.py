import sys
sys.path.append("../lib")

import numpy
import pandas

import nbsupport.tcga

segments = {}

for segment, genes in nbsupport.tcga.read_gistic_output("../data/tcga/del_genes.conf_95.pancan12.txt").iteritems():
    segments["_".join([segment, "loss"])] = genes

for segment, genes in nbsupport.tcga.read_gistic_output("../data/tcga/amp_genes.conf_95.pancan12.txt").iteritems():
    segments["_".join([segment, "gain"])] = genes

cancer_genes = pandas.read_table("../data/tcga/cancer-genes.tsv")

entrez_gene_info = pandas.read_table("../data/entrez/seq_gene.md.gz", usecols=[1, 9, 12], compression="gzip", low_memory=False)
entrez_gene_info = entrez_gene_info[entrez_gene_info.group_label == "GRCh37.p13-Primary Assembly"]

mut_genes = pandas.read_csv("../data/tcga/mutational-drivers.csv")

high_conf_drivers = mut_genes["Gene Symbol"][mut_genes["Putative Driver Category"] == "High Confidence Driver"]

mut_drivers = pandas.DataFrame.from_items([
        ("gene", high_conf_drivers),
        ("chrom", numpy.r_[entrez_gene_info.chromosome.values, numpy.nan][pandas.match(high_conf_drivers, entrez_gene_info.feature_name)]),
        ("type", "mut")])

selected_genes = numpy.union1d(
    numpy.union1d(
        numpy.intersect1d(numpy.concatenate(segments.values()), mut_genes["Gene Symbol"]),
        numpy.intersect1d(numpy.concatenate(segments.values()), cancer_genes.symbol)),
    [g[0].strip("[]") for g in segments.itervalues() if len(g) == 1])

rows = []
for gene in selected_genes:
    segment = next(seg for seg, genes in segments.iteritems() if gene in map(lambda s: s.strip("[]"), genes))
    rows.append((gene, segment[:max(segment.find("p"), segment.find("q"))],  segment.rsplit("_")[-1]))

cn_drivers = pandas.DataFrame(rows, columns=["gene", "chrom", "type"])

cn_drivers = cn_drivers[cn_drivers.gene != "DUX4"]

drivers = pandas.concat([cn_drivers, mut_drivers]).sort_values("gene")

drivers.reset_index(drop=True, inplace=True)

gene2chrom = {
    "AKD1": "6",
    "MLL": "11",
    "MLL2": "12",
    "MLL3": "7"
}

for i, row in drivers[drivers.chrom.isnull()].iterrows():
    drivers.chrom[i] = gene2chrom[row.gene]

drivers.to_csv("../data/tcga/selected-genes.txt", sep="\t", index=False, na_rep="NA")

