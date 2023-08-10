get_ipython().run_cell_magic('capture', '', '%run 01-download.ipynb')

get_ipython().run_cell_magic('capture', '', '%run 02-transform.ipynb')

# Import Python tools
import calculate
import pandas as pd

get_ipython().magic('matplotlib inline')

# Read in state-level data
state_df = pd.read_csv("./data/transformed_state.csv", dtype={"area_fips": "str"})

state_df.head()

# Filter that down to just California crop workers
ca_state_df = state_df[state_df.area_fips.str.startswith("06")]

ca_state_crops = ca_state_df[ca_state_df.industry_group == 'crops'].set_index("year")

ca_state_crops.at[2015, "avg_annual_pay_2015"]

ca_state_crops.at[2010, "avg_annual_pay_2015"]

calculate.percentage_change(
    ca_state_crops.at[2010, "avg_annual_pay_2015"],
    ca_state_crops.at[2015, "avg_annual_pay_2015"]
)

ca_state_overall = ca_state_df[ca_state_df.industry_group == 'total'].set_index("year")

ca_state_overall.at[2010, "avg_annual_pay_2015"]

ca_state_overall.at[2015, "avg_annual_pay_2015"]

calculate.percentage_change(
    ca_state_overall.at[2010, "avg_annual_pay_2015"],
    ca_state_overall.at[2015, "avg_annual_pay_2015"]
)

ca_state_crops.reset_index().plot(kind='line', x='year', y='avg_annual_pay_2015', figsize=(10, 6))

ca_state_crops.reset_index()[[
    'year',
    'avg_annual_pay_2015'
]]

ca_state_crops.reset_index()[[
    'year',
    'avg_annual_pay_2015'
]].to_csv("./data/crops-wages-by-year.csv", index=False)

# Read in county-level data
county_df = pd.read_csv("./data/transformed_county.csv", dtype={"area_fips": str})

county_df.head()

# Filter it down to crops
county_crops = county_df[county_df.industry_group == 'crops']

# Filter it down to the latest year of data
trimmed_county_crops_2015 = county_crops[county_crops.year==2015]

# Filter it down to California
trimmed_california_county_crops = trimmed_county_crops_2015[trimmed_county_crops_2015.area_fips.str.startswith("06")]

trimmed_california_county_crops.sort_values("avg_annual_pay_2015", ascending=False).head(60)

trimmed_california_county_crops.sort_values("avg_annual_pay_2015", ascending=False).to_csv("./data/map.csv", index=False)

