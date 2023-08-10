import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_unweighted
import seaborn
get_ipython().magic('pylab inline')
pd.set_option('display.max_rows', 150)

vep_cols = """Allele|Annotation|Impact|Gene_name|Gene_id|Feature_type|Feature_id|Biotype|EXON|INTRON|HGVSc|HGVSp|cDNA_position|CDS_position|Protein_position|Amino_acids|Codons|Existing_variation|DISTANCE|STRAND|FLAGS|SYMBOL_SOURCE|HGNC_ID|CANONICAL|CCDS|HGVS_OFFSET"""
vep_cols = vep_cols.split("|")
vep_cols = [term.strip().capitalize() for term in vep_cols]

vep_df = pd.read_table('./cftr.grch37.vep.vcf', header=0, skiprows=6, usecols=range(10))
info_df = vep_df["INFO"].str.replace("ANN=", "").str.split(",").apply(pd.Series, 1).stack()
info_df = info_df.str.split("|").apply(pd.Series, 1)
info_df.index = info_df.index.droplevel(-1)
info_df.columns = vep_cols
vep_df = vep_df.join(info_df)
vep_df = vep_df[['POS', 'REF', 'ALT', 'Feature_id', 'Biotype', 'Annotation', 'Impact', 'Hgvsc']]
del info_df
vep_df.head()

snpeff_cols = """Allele|Annotation|Impact|Gene_name|Gene_ID|Feature_type|Feature_ID|Biotype|Rank|HGVSc|HGVSp|cDNA_position|CDS_position|Protein_position|Distance|Errors"""
snpeff_cols = snpeff_cols.split("|")
snpeff_cols = [term.strip().capitalize() for term in snpeff_cols]

snpeff_df = pd.read_table('./cftr.grch37.snpeff.vcf', header=0, skiprows=9, usecols=range(10))
info_df = snpeff_df["INFO"].str.split(";").apply(pd.Series, 1)[0] #snpeff includes two other INFO fields that we don't need
info_df = info_df.str.replace("ANN=", "").str.split(",").apply(pd.Series, 1).stack()
info_df = info_df.str.split("|").apply(pd.Series, 1)
info_df.index = info_df.index.droplevel(-1)
info_df.columns = snpeff_cols
snpeff_df = snpeff_df.join(info_df)
snpeff_df = snpeff_df[['POS', 'REF', 'ALT', 'Feature_id', 'Biotype', 'Annotation', 'Impact', 'Hgvsc']]
del info_df
snpeff_df.head()

#From http://snpeff.sourceforge.net/VCFannotationformat_v1.0.pdf with additions:
#VEP: non_coding_transcript_exon_variant, non_coding_transcript_variant, protein altering variant, 
#incomplete_terminal_codon_variant, NMD_transcript_variant
#Snpeff: conservative_inframe_deletion, conservative_inframe_insertion, structural_interaction_variant, 5_prime_UTR_truncation

ranked_terms = ["chromosome_number_variation","exon_loss_variant","frameshift_variant","stop_gained","stop_lost",
                "start_lost","splice_acceptor_variant","splice_donor_variant","rare_amino_acid_variant","missense_variant",
                "inframe_insertion","conservative_inframe_insertion", "disruptive_inframe_insertion","inframe_deletion","conservative_inframe_deletion", "disruptive_inframe_deletion",
                "5_prime_UTR_truncation+exon_loss_variant","5_prime_UTR_truncation","3_prime_UTR_truncation+exon_loss","splice_branch_variant",
                "splice_region_variant","stop_retained_variant","initiator_codon_variant",
                "synonymous_variant","initiator_codon_variant+non_canonical_start_codon","stop_retained_variant",
                "5_prime_UTR_variant","3_prime_UTR_variant","5_prime_UTR_premature_start_codon_gain_variant",
                "structural_interaction_variant", "protein_altering_variant","upstream_gene_variant","downstream_gene_variant",
                "TF_binding_site_variant","regulatory_region_variant","miRNA","custom","sequence_feature",
                "conserved_intron_variant","intron_variant","intragenic_variant","conserved_intergenic_variant",
                "intergenic_region", "coding_sequence_variant", "non_coding_exon_variant","non_coding_transcript_exon_variant",
                "nc_transcript_variant","non_coding_transcript_variant","NMD_transcript_variant", "incomplete_terminal_codon_variant", "gene_variant","chromosome"]
def term_rank(term):
    return ranked_terms.index(term)

vep_df["Effect"] = vep_df.apply(lambda row: min(row["Annotation"].split("&"), key=term_rank), axis=1)
snpeff_df["Effect"] = snpeff_df.apply(lambda row: min(row["Annotation"].split('&'), key=term_rank), axis=1)

vc_vep = vep_df.groupby(['Effect']).size()
vc_vep.name = "VEP"
vc_snpeff = snpeff_df.groupby(['Effect']).size()
vc_snpeff.name = "SnpEff"
vc_df = pd.DataFrame([vc_vep, vc_snpeff])
print("Annotations\n")
print(vc_df.transpose().fillna(0))
impact_vep = vep_df.groupby(['Impact']).size()
impact_vep.name = "VEP"
impact_snpeff = snpeff_df.groupby(['Impact']).size()
impact_snpeff.name = "SnpEff"
impact_df = pd.DataFrame([impact_vep, impact_snpeff])
print("\nImpacts")
print(impact_df.transpose())
counts_vep = vep_df.count()
counts_vep.name = 'VEP'
counts_snpeff = snpeff_df.count()
counts_snpeff.name = 'SnpEff'
counts_df = pd.DataFrame([counts_vep, counts_snpeff])
print("\nCounts")
print(counts_df.transpose())

vep_df[vep_df['Effect'].str.contains('protein_altering_variant')][:1]

vep_df.ix[19511]

snpeff_df.ix[19511]

snpeff_df[snpeff_df['Effect'].str.contains('structural_interaction_variant')][:1]

vep_df.ix[58085]

snpeff_df.ix[58085]

snpeff_df[snpeff_df['Effect'].str.contains('sequence_feature')][3:4]

vep_df.ix[19449]

snpeff_df.ix[19449]

vep_df[vep_df['Effect'].str.contains('coding_sequence_variant')][:4]

vep_df.ix[32771]

snpeff_df.ix[32771]

snpeff_df = snpeff_df[~snpeff_df['Effect'].str.contains('structural_interaction_variant|sequence_feature')]

collapse_map = {
'3_prime_UTR_variant': '3_prime_UTR_variant', 
'5_prime_UTR_premature_start_codon_gain_variant': '5_prime_UTR_premature_start_codon_gain_variant',
'5_prime_UTR_variant': '5_prime_UTR_variant',
'coding_sequence_variant': 'coding_sequence_variant',
'conservative_inframe_deletion': 'inframe_deletion',
'conservative_inframe_insertion': 'inframe_insertion',
'disruptive_inframe_deletion': 'inframe_deletion',
'disruptive_inframe_insertion': 'inframe_insertion',
'downstream_gene_variant': 'downstream_gene_variant',
'exon_loss_variant': 'exon_loss_variant',
'frameshift_variant': 'frameshift_variant',
'incomplete_terminal_codon_variant': 'incomplete_terminal_codon_variant',
'inframe_deletion': 'inframe_deletion',
'inframe_insertion': 'inframe_insertion', 
'initiator_codon_variant': 'initiator_codon_variant',
'intergenic_region': 'intergenic_region',
'intron_variant': 'intron_variant',
'missense_variant': 'missense_variant',
'non_coding_transcript_exon_variant': 'non_coding_transcript_exon_variant',
'non_coding_transcript_variant': 'non_coding_transcript_variant',
'protein_altering_variant': 'inframe_insertion',
'splice_acceptor_variant': 'splice_acceptor_variant',
'splice_donor_variant': 'splice_donor_variant',
'splice_region_variant': 'splice_region_variant',
'start_lost': 'start_lost',
'stop_gained': 'stop_gained',
'stop_lost': 'stop_lost', 
'stop_retained_variant': 'stop_retained_variant',
'synonymous_variant': 'synonymous_variant',
'upstream_gene_variant': 'upstream_gene_variant'}

vep_df['Normalized_effect'] = vep_df['Effect'].apply(lambda eff: collapse_map[eff])
snpeff_df['Normalized_effect'] = snpeff_df['Effect'].apply(lambda eff: collapse_map[eff])

vc_vep = vep_df.groupby(['Normalized_effect']).size()
vc_vep.name = "VEP"
vc_snpeff = snpeff_df.groupby(['Normalized_effect']).size()
vc_snpeff.name = "SnpEff"
vc_df = pd.DataFrame([vc_vep, vc_snpeff])
print("Annotations\n")
print(vc_df.transpose().fillna(0))
vc_df.transpose().plot(kind="barh", fontsize=13, figsize=(16,8))

effect_df = pd.merge(vep_df, snpeff_df, on=['POS', 'REF', 'ALT', "Feature_id" ], how='outer', suffixes=('_vep','_snpeff'))

effect_df['Impact_match'] = effect_df.apply(lambda row: row['Impact_vep'] == row['Impact_snpeff'], axis=1)

effect_df['Effect_match'] = effect_df.apply(lambda row: row['Effect_vep'] == row['Effect_snpeff'], axis=1)

effect_df['Normalized_Effect_match'] = effect_df.apply(lambda row: row['Normalized_effect_vep'] == row['Normalized_effect_snpeff'], axis=1)

round(effect_df['Impact_match'].value_counts()/effect_df['Impact_match'].size*100, 2)

effect_df.groupby(['Impact_vep', 'Impact_snpeff'])['Impact_match'].count()

figure, axes = plt.subplots(2, 2)
figure.set_size_inches(10,10)
figure.suptitle("VEP-SnpEff Impact Concordance", fontsize=14, fontweight='bold')
for idx, level in enumerate(effect_df['Impact_snpeff'].dropna().unique()):
    pos = (int(idx/2), idx%2)
    vep_level = effect_df[effect_df['Impact_vep'] == level]
    snpeff_level = effect_df[effect_df['Impact_snpeff'] == level]
    axes[pos].set_title(level, fontsize=12, fontweight='bold')
    venn2_unweighted([set(vep_level.index.values), set(snpeff_level.index.values)], set_labels=('VEP', 'SnpEff'), set_colors=('g', 'b'),
                     ax=axes[pos[0]][pos[1]])
plt.tight_layout()
plt.subplots_adjust(top=.95)
plt.show()

round(effect_df['Effect_match'].value_counts()/effect_df['Effect_match'].size*100, 2)

pd.DataFrame(effect_df.groupby(['Effect_vep', 'Effect_snpeff'])['Effect_match'].count())

round(effect_df['Normalized_Effect_match'].value_counts()/effect_df['Normalized_Effect_match'].size*100, 2)

pd.DataFrame(effect_df.groupby(['Normalized_effect_vep', 'Normalized_effect_snpeff'])['Effect_match'].count())

effect_list = sorted(set(effect_df['Normalized_effect_vep'].dropna().unique()) | set(effect_df['Normalized_effect_snpeff'].dropna().unique()), 
key=term_rank)

figure, axes = plt.subplots(13, 2)
figure.set_size_inches(10,45)
figure.suptitle("VEP-SnpEff Effect Concordance", fontsize=14, fontweight='bold')

for idx, effect in enumerate(effect_list):
    pos = (int(idx/2), idx%2)
    vep_effect = effect_df[effect_df['Normalized_effect_vep'] == effect]
    snpeff_effect = effect_df[effect_df['Normalized_effect_snpeff'] == effect]
    axes[pos].set_title(effect, fontsize=12, fontweight='bold')
    venn2_unweighted([set(vep_effect.index.values), set(snpeff_effect.index.values)], set_labels=('VEP', 'SnpEff'), set_colors=('g', 'b'),
                     ax=axes[pos[0]][pos[1]])
axes[12, 1].axis('off')

plt.subplots_adjust(top=.95)
plt.show()

effect_df['Hgvsc_vep'] = effect_df['Hgvsc_vep'].str.replace(r'^.*:', '').str.strip()
effect_df['Hgvsc_snpeff'] = effect_df['Hgvsc_snpeff'].str.replace(r'^.*:', '').str.strip()

effect_df['Hgvsc_match'] = effect_df.apply(lambda row: row['Hgvsc_vep'] == row['Hgvsc_snpeff'], axis=1)

round(effect_df['Hgvsc_match'].value_counts()/effect_df['Hgvsc_match'].size*100, 2)

has_hgvsc = (effect_df['Hgvsc_vep'] != '') & (effect_df['Hgvsc_snpeff'] != '')

filtered = effect_df[has_hgvsc]
round(filtered['Hgvsc_match'].value_counts()/filtered['Hgvsc_match'].size*100, 2)

is_coding = (effect_df['Biotype_vep'] == 'protein_coding') & (effect_df['Biotype_snpeff'] == 'protein_coding')

filtered_2 = effect_df[has_hgvsc & is_coding]

round(filtered_2['Hgvsc_match'].value_counts()/filtered_2['Hgvsc_match'].size*100, 2)

on_canonical = (effect_df['Feature_id'] == "ENST00000003084")

filtered_3 = effect_df[has_hgvsc & on_canonical]

round(filtered_3['Hgvsc_match'].value_counts()/filtered_3['Hgvsc_match'].size*100, 2)

mismatch_df = filtered_2[filtered_2['Hgvsc_match'] == False]
mismatch_df[['POS', 'REF', 'ALT', 'Feature_id', 'Hgvsc_vep', 'Hgvsc_snpeff', 'Effect_vep', 'Effect_snpeff']]

round(mismatch_df['Effect_match'].value_counts()/mismatch_df['Effect_match'].size*100, 2)

vc_vep = mismatch_df.groupby(['Effect_vep']).size()
vc_vep.name = "VEP"
vc_snpeff = mismatch_df.groupby(['Effect_snpeff']).size()
vc_snpeff.name = "SnpEff"
vc_df = pd.DataFrame([vc_vep, vc_snpeff])
print("Annotations\n")
print(vc_df.transpose().fillna(0))

def create_vcf_header():
    metadata_lines = [
        '##fileformat=VCFv4.0',
        '##INFO=<ID=TX,Number=1,Type=String,Description="Corresponding Transcript"',
        '##INFO=<ID=VE,Number=1,Type=String,Description="VEP Effect"',
        '##INFO=<ID=VI,Number=1,Type=String,Description="Vep Impact"',
        '##INFO=<ID=VC,Number=1,Type=String,Description="Vep coding HGVS"',
        '##INFO=<ID=SE,Number=1,Type=String,Description="SnpEff Effect"',
        '##INFO=<ID=SI,Number=1,Type=String,Description="SnpEff Impact"',
        '##INFO=<ID=SC,Number=1,Type=String,Description="SnpEff coding HGVS"']
    header_fields = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']
    metadata_lines.append('\t'.join(header_fields))
    vcf_header = '\n'.join(metadata_lines)
    return vcf_header

def create_vcf_line(row):
    line = []
    line.append('7')
    line.append(row['POS'])
    line.append('.')
    line.extend(row[['REF', 'ALT']])
    line.append('.')
    line.append('PASS')
    info = 'TX={};VE={};VI={};VC={};SE={};SI={};SC={}'.format(*row[['Feature_id', 'Effect_vep', 'Impact_vep', 'Hgvsc_vep',
                                                                    'Effect_snpeff', 'Impact_snpeff', 'Hgvsc_snpeff']])
    line.append(info)
    line.append('TX:VE:VI:VC:SE:SI:SC')
    return '\t'.join(map(str, line))

def create_vcf(df, idxs, filename):
    with open(filename, "w") as vcf_out:
        vcf_out.write(create_vcf_header() + '\n')
        for i in idxs:
            vcf_out.write(create_vcf_line(df.ix[i]) + '\n')

hgvs_mismatch = (effect_df['Hgvsc_vep'] != '') & (effect_df['Hgvsc_snpeff'] != effect_df['Hgvsc_vep'])
hgvs_mismatch_idxs = effect_df[hgvs_mismatch].sort_values(by=['POS']).index.tolist()

impact_mismatch_idxs = effect_df[~effect_df['Impact_match']].sort_values(by=['POS']).index.tolist()
effect_mismatch_idxs = effect_df[~effect_df['Effect_match']].sort_values(by=['POS']).index.tolist()

create_vcf(effect_df, effect_mismatch_idxs, 'effect_mismatch.vcf')
create_vcf(effect_df, impact_mismatch_idxs, 'impact_mismatch.vcf')
create_vcf(effect_df, hgvs_mismatch_idxs, 'hgvs_mismatch.vcf')

