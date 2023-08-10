import pandas as pd

# Create an example dataframe
data = {'name': ['Jason', 'Molly'], 
        'country': [['Syria', 'Lebanon'],['Spain', 'Morocco']]}
df = pd.DataFrame(data)
df

df[df['country'].map(lambda country: 'Syria' in country)]

