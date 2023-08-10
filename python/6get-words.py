import pandas as pd
df=pd.read_csv('data/train-clean.csv',sep="\t")
df = df.drop('Unnamed: 0', 1)
df.head()

mask=df.Phrase.str.match(r'\A[\w-]+\Z')
df=df[mask]
df.head()

df.to_csv("data/words.csv", sep='\t')

print("done successfully")

