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

alleles = ['A0101', 'A0201', 'A0203', 'A0204', 'A0207', 'A0301', 'A2402', 'A2902', 'A3101', 'A6802', 'B3501', 
           'B4402', 'B4403', 'B5101', 'B5401', 'B5701']
proper_alleles = ['HLA-{0}:{1}'.format(x[:3], x[3:]) for x in alleles]

observed = []
for allele in alleles:
    tmp_df = pd.read_csv(PATH_TO_DATA + '/data/MS_validation/single_allele/{0}.best_rank.csv'.format(allele), index_col=0)
    tmp_df.columns = ['score']
    observed.extend(tmp_df.score)
observed = pd.Series(observed)

random_df = pd.read_csv(PATH_TO_DATA + '/data/random_allele_matrix.csv', index_col=0)
random = []
for allele in proper_alleles:
    random.extend(random_df[allele])
random = pd.Series(random)

TP, FP = [], []
for cutoff in [0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    TP.append((observed < cutoff).mean())
    FP.append((random < cutoff).mean())

plt.plot(FP, TP, c='k', label='{0}: {1}'.format('Best Rank', str(round(metrics.auc(FP, TP), 2))))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig(PATH_TO_GENERATED_FIGURES + 'ROC.BR.pdf')

expression = pd.read_csv(PATH_TO_DATA + '/data/cell_line_gene_expression.csv', index_col=0)
expression['gene'] = expression.index

def get_gene(x):
    return x.split('_')[0]

expression.head()

cell_lines = ['A2780', 'ov90', 'HeLa', 'skov3', 'A375']
observed_totals, predicted_totals = [], []
for i, cell in enumerate(cell_lines):
    # cell line affinities; filtering (common, ensembl) already complete
    df = pd.read_csv(PATH_TO_DATA + '/data/MS_validation/mutations/{0}.csv'.format(cell), index_col=0)
    df['gene'] = df['mutation'].apply(get_gene)
    
    # reduce to expressed
    df = pd.merge(df, expression, on='gene', how='inner')
    df = df[df[cell] > df[cell].quantile(0.25)]
    
    # strong
    strong = (df.BR < 0.5)
    predicted_totals.append(len(df[strong]))
    observed_totals.append(len(df[(strong)&(df.observed)]))
    
    # weak
    weak = ((df.BR >= 0.5) & (df.BR < 2))
    predicted_totals.append(len(df[weak]))
    observed_totals.append(len(df[(weak)&(df.observed)]))

results_df = pd.DataFrame({'Cell_line': ['A2780', 'A2780', 'OV90', 'OV90', 'HeLa', 'HeLa','SKOV3', 'SKOV3', 'A375', 'A375'],
                           'Binding Strength': ['Strong', 'Weak']*5,
                           'Observed': observed_totals,
                           'Predicted': predicted_totals})
results_df['Fraction_observed'] = results_df.Observed / results_df.Predicted

sns.barplot(x='Cell_line', y='Fraction_observed', hue='Binding Strength', data=results_df, color='white')
plt.xlabel('Cell line')
plt.ylabel('Fraction of mutations observed')
plt.savefig(PATH_TO_GENERATED_FIGURES + 'MS_validation.mutations.pdf')



