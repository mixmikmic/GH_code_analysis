import os
import settings
import pandas as pd
from cpi import to_2015_dollars

df = pd.read_excel(
    os.path.join(settings.input_dir, "avgprice_annual.xlsx"),
    skiprows=1
)

california_vs_usa_df = df[
    df['State'].isin(["CA", "US"]) &
    (df['Industry Sector Category'] == 'Total Electric Industry')
].sort_values(["State", "Year"])

california_vs_usa_df['Total (2015 dollars)'] = california_vs_usa_df.apply(
    lambda x: to_2015_dollars(x['Total'], str(int(x['Year']))),
    axis=1
)

california_vs_usa_df.to_csv(os.path.join(settings.output_dir, "annual-price-california-vs-usa.csv"), index=False)

