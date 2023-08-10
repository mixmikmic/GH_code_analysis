import os
import settings
import pandas as pd

df = pd.read_excel(
    os.path.join(settings.input_dir, 'sales_annual.xlsx'),
    skiprows=1,
    index_col="Year"
)

california_df = df[
    (df['State'] == 'CA') &
    (df['Industry Sector Category'] == 'Total Electric Industry')
].sort_index()

california_df['Business'] = california_df.apply(lambda x: x['Total'] - x['Residential'], axis=1)

california_df.to_csv(os.path.join(settings.output_dir, "annual-sales-california.csv"))

