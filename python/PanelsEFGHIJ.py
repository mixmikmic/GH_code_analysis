import io
from IPython.nbformat import current
def execute_notebook(nbfile):
    with io.open(nbfile) as f:
        nb = current.read(f, 'json')
    ip = get_ipython()
    for cell in nb.worksheets[0].cells:
        if cell.cell_type != 'code':
            continue
        ip.run_cell(cell.input)
execute_notebook("../../imports/imports.ipynb")

df = pd.read_csv(PATH_TO_DATA + 'data/hla_typing/hla_types.csv', index_col=0)

combined_df = pd.DataFrame({'A': list(df.A1) + list(df.A2),
                            'B': list(df.B1) + list(df.B2),
                            'C': list(df.C1) + list(df.C2)})

df_A = pd.DataFrame(combined_df.A.value_counts()/len(combined_df)).reset_index()[:15]
df_A.columns = ['allele', 'frequency']
sns.set_color_codes("muted")
sns.barplot(x="frequency", y="allele", data=df_A,
            label="Total", color="white")
plt.xlabel('Allele Frequency in TCGA')

df_B = pd.DataFrame(combined_df.B.value_counts()/len(combined_df)).reset_index()[:15]
df_B.columns = ['allele', 'frequency']
sns.set_color_codes("muted")
sns.barplot(x="frequency", y="allele", data=df_B,
            label="Total", color="white")
plt.xlabel('Allele Frequency in TCGA')

df_C = pd.DataFrame(combined_df.C.value_counts()/len(combined_df)).reset_index()[:15]
df_C.columns = ['allele', 'frequency']
sns.set_color_codes("muted")
sns.barplot(x="frequency", y="allele", data=df_C,
            label="Total", color="white")
plt.xlabel('Allele Frequency in TCGA')

clinical = pd.read_csv(PATH_TO_DATA + 'data/clinical/ancestory.csv', index_col=0)

asian = list(clinical[clinical.race.isin(['ASIAN'])].index)
black = list(clinical[clinical.race.isin(['BLACK OR AFRICAN AMERICAN'])].index)
white = list(clinical[clinical.race.isin(['WHITE'])].index)

combined_df_asian = pd.DataFrame({'A': list(df[df.Sample.isin(asian)].A1) + list(df[df.Sample.isin(asian)].A2),
                            'B': list(df[df.Sample.isin(asian)].B1) + list(df[df.Sample.isin(asian)].B2),
                            'C': list(df[df.Sample.isin(asian)].C1) + list(df[df.Sample.isin(asian)].C2)})
combined_df_white = pd.DataFrame({'A': list(df[df.Sample.isin(white)].A1) + list(df[df.Sample.isin(white)].A2),
                            'B': list(df[df.Sample.isin(white)].B1) + list(df[df.Sample.isin(white)].B2),
                            'C': list(df[df.Sample.isin(white)].C1) + list(df[df.Sample.isin(white)].C2)})
combined_df_black = pd.DataFrame({'A': list(df[df.Sample.isin(black)].A1) + list(df[df.Sample.isin(black)].A2),
                            'B': list(df[df.Sample.isin(black)].B1) + list(df[df.Sample.isin(black)].B2),
                            'C': list(df[df.Sample.isin(black)].C1) + list(df[df.Sample.isin(black)].C2)})

# asian
all_dfs = []
for gene in ['A', 'B', 'C']:
    tmp = pd.DataFrame(combined_df_asian[gene].value_counts() / len(combined_df_asian)).reset_index()
    tmp.columns = ['allele', 'asian_tcga_frequency']
    all_dfs.append(tmp)
asian_tcga = pd.concat(all_dfs)
# white
all_dfs = []
for gene in ['A', 'B', 'C']:
    tmp = pd.DataFrame(combined_df_white[gene].value_counts() / len(combined_df_white)).reset_index()
    tmp.columns = ['allele', 'white_tcga_frequency']
    all_dfs.append(tmp)
white_tcga = pd.concat(all_dfs)
# black
all_dfs = []
for gene in ['A', 'B', 'C']:
    tmp = pd.DataFrame(combined_df_black[gene].value_counts() / len(combined_df_black)).reset_index()
    tmp.columns = ['allele', 'black_tcga_frequency']
    all_dfs.append(tmp)
black_tcga = pd.concat(all_dfs)

def restrict_alleles(x):
    if ('A*' in x) | ('B*' in x) | ('C*' in x):
        return True
    else:
        return False

african = pd.read_csv(PATH_TO_DATA + 'data/world_populations/usa_nmdp_african', sep='\t', header=None)
african = african[[1, 4]]
african.columns = ['allele', 'african_frequency']
african = african[african['allele'].apply(restrict_alleles)]

caucasian = pd.read_csv(PATH_TO_DATA + 'data/world_populations/usa_nmdp_caucasian', sep='\t', header=None)
caucasian = caucasian[[1, 4]]
caucasian.columns = ['allele', 'caucasian_frequency']
caucasian = caucasian[caucasian['allele'].apply(restrict_alleles)]

japanese = pd.read_csv(PATH_TO_DATA + 'data/world_populations/usa_nmdp_japanese', sep='\t', header=None)
japanese = japanese[[1, 4]]
japanese.columns = ['allele', 'japanese_frequency']
japanese = japanese[japanese['allele'].apply(restrict_alleles)]

df = pd.merge(pd.merge(white_tcga, pd.merge(asian_tcga, black_tcga, on='allele', how='outer'),
                       on='allele', how='outer'),
            pd.merge(pd.merge(african, caucasian, on='allele', how='outer'),
               japanese, on='allele', how='outer'), on='allele', how='outer')
df = df.fillna(0)

colors = sns.color_palette('Set2')

plt.scatter(df.asian_tcga_frequency, df.african_frequency, c=colors[0], label='TCGA Asian - {0:.2f}'.format(sp.pearsonr(df.asian_tcga_frequency, df.african_frequency)[0]))
plt.scatter(df.white_tcga_frequency, df.african_frequency, c=colors[1], label='TCGA Caucasian - {0:.2f}'.format(sp.pearsonr(df.white_tcga_frequency, df.african_frequency)[0]))
plt.scatter(df.black_tcga_frequency, df.african_frequency, c=colors[2], label='TCGA African - {0:.2f}'.format(sp.pearsonr(df.black_tcga_frequency, df.african_frequency)[0]))
plt.ylim(0, .35)
plt.xlim(0, .35)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.xlabel('TCGA Frequencies')
plt.ylabel('Alternate African Frequency')

plt.scatter(df.white_tcga_frequency, df.japanese_frequency, c=colors[1], label='TCGA Caucasian - {0:.2f}'.format(sp.pearsonr(df.white_tcga_frequency, df.japanese_frequency)[0]))
plt.scatter(df.black_tcga_frequency, df.japanese_frequency, c=colors[2], label='TCGA African - {0:.2f}'.format(sp.pearsonr(df.black_tcga_frequency, df.japanese_frequency)[0]))
plt.scatter(df.asian_tcga_frequency, df.japanese_frequency, c=colors[0], label='TCGA Asian - {0:.2f}'.format(sp.pearsonr(df.asian_tcga_frequency, df.japanese_frequency)[0]))
plt.ylim(0, .35)
plt.xlim(0, .35)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('TCGA Frequencies')
plt.ylabel('Alternate Japanese Frequency')

plt.scatter(df.asian_tcga_frequency, df.caucasian_frequency, c=colors[0], label='TCGA Asian - {0:.2f}'.format(sp.pearsonr(df.asian_tcga_frequency, df.caucasian_frequency)[0]))
plt.scatter(df.black_tcga_frequency, df.caucasian_frequency, c=colors[2], label='TCGA African - {0:.2f}'.format(sp.pearsonr(df.black_tcga_frequency, df.caucasian_frequency)[0]))
plt.scatter(df.white_tcga_frequency, df.caucasian_frequency, c=colors[1], label='TCGA Caucasian - {0:.2f}'.format(sp.pearsonr(df.white_tcga_frequency, df.caucasian_frequency)[0]))
plt.ylim(0, .35)
plt.xlim(0, .35)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.xlabel('TCGA Frequencies')
plt.ylabel('Alternate Caucasian Frequency')



