output_file = "../data/cleaned_coalpublic2013.csv"

import numpy as np
import pandas as pd

df = pd.read_excel("../data/coalpublic2013.xls", header=2, index_col='MSHA ID')

# Mistake, renaming Indepedent to Independent
df['Company Type'].unique()

df['Company Type'].replace(to_replace='Indepedent Producer Operator', 
                           value='Independent Producer Operator',
                           inplace=True)

# Changing spaces to _
df.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)

# We are removing data here!
# Coal mines without ANY coal production are removed.
df = df[df['Production_(short_tons)'] > 0]

len(df)

df['log_production'] = np.log(df['Production_(short_tons)'])

df.to_csv(output_file)



