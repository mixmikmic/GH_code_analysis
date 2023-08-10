import os
import settings
import pandas as pd

df = pd.read_excel(
    os.path.join(settings.input_dir, "existcapacity_annual.xlsx"),
    na_values=[' ', '.'],
    index_col='Year'
)

california_df = df[
    (df['State Code'] == 'CA') &
    (df['Producer Type'] == 'Total Electric Power Industry') &
    (df['Fuel Source'] == 'All Sources')
]

california_df.to_csv(os.path.join(settings.output_dir, "annual-capacity-california.csv"))

