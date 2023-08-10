def vcf_to_maf(df, filter_multiple=True):
    import pandas as pd
    
    df['POS'] = df['POS'].astype(int)
    if filter_multiple: 
        df = df[df['ALT'].apply(lambda x: len(x) == 1)]
        df['ALT'] = df['ALT'].apply(lambda x: x[0])
        
    df = df[['CHROM', 'POS', 'REF', 'ALT']].rename(columns={
        'CHROM': 'chr',
        'POS':'start', 
        'REF':'ref_allele', 
        'ALT':'alt_allele'
    })
    
    ## SNP
    ## Start: POS
    ## End: POS
    ## Ref = Ref
    ## Alt = Alt
    snp = df[df['ref_allele'].apply(lambda x: len(x)) == df['alt_allele'].apply(lambda x: len(x))]
    snp['end'] = snp['start']

    ## Insertion: 
    ## Start: POS
    ## End: POS+1
    ## Ref: - 
    ## Alt: Alt[1:]
    ins = df[df['ref_allele'].apply(lambda x: len(x)) < df['alt_allele'].apply(lambda x: len(x))]
    ins['start'] = ins['start']
    ins['end'] = ins['start'] + 1
    ins['ref_allele'] = "-"
    ins['alt_allele'] = ins['alt_allele'].apply(lambda x: str(x)[1:])

    ## Deletion: 
    ## Start: POS+1
    ## End: POS+len(Ref)-1
    ## Ref: Ref[1:]
    ## Alt: - 
    dels = df[df['ref_allele'].apply(lambda x: len(x)) > df['alt_allele'].apply(lambda x: len(x))]
    dels['start'] = dels['start'] + 1
    dels['end'] = dels['start'] + dels['ref_allele'].apply(lambda x: len(x)-2) 
    dels['ref_allele'] = dels['ref_allele'].apply(lambda x: str(x)[1:])
    dels['alt_allele'] = '_'
    
    maf_cols = ['chr', 'start', 'end', 'ref_allele', 'alt_allele']

    return pd.concat([snp, ins, dels], axis=0)[maf_cols]

import pickle
import pandas as pd
cancerInputFile = '../data/VCF_Data_Cancer'
with open(cancerInputFile, "rb") as f: 
    cancerData = pickle.load(f)
cancerData = pd.read_csv('../data/VCF_300.csv')
cancerMaf = vcf_to_maf(cancerData, filter_multiple=True)
cancerMaf.to_csv('../data/cancer_G97552_maflite.txt', sep='\t', index=False)
delete('cancerMaf')

normalInputFile = '../data/VCF_Data_Normal'

with open(normalInputFile, "rb") as g: 
    normalData = pickle.load(g)
    
normalMaf = vcf_to_maf(normalData, filter_multiple=True)
normalMaf.to_csv('../data/normal_G91716_maflite.txt', sep='\t', index=False)
delete('normalMaf')



