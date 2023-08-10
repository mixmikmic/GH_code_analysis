import pandas as pd

prescription = pd.read_csv('data/prescriptions_sample.csv.gz', compression='gzip')
prescription.columns

len(prescription)

prescription[['medication','target']].groupby(['medication']).agg(['count'])



