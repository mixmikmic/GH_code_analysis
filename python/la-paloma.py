import os
import pandas as pd

get_ipython().magic('matplotlib inline')

pd.options.display.float_format = "{:,.2f}".format

data_dir = os.path.join(os.getcwd(), 'data', 'output')

latimes_df = pd.read_csv(
    os.path.join(data_dir, "plants-california.csv"),
)

trimmed_latimes_columns = [
    'plant_id',
    'plant_name_gen',
    'year',
    'net_generation_mwh',
    'capacity_mwh',
    'capacity_utilization'
]

trimmed_latimes_df = latimes_df[trimmed_latimes_columns]

trimmed_latimes_df[
    (trimmed_latimes_df.year > 2005)
].groupby([
    'plant_id',
    'plant_name_gen'
]).net_generation_mwh.sum().reset_index().sort_values(
    "net_generation_mwh",
    ascending=False
).head(10)

eia_df = pd.read_excel(
    "data/input/EIA923_Schedules_2_3_4_5_M_12_2016.xlsx",
    sheetname='Page 1 Generation and Fuel Data',
    skiprows=5
)

trimmed_eia_columns = [
    'Plant Id',
    'Plant Name',
    'Plant State',
    'Net Generation\n(Megawatthours)'
]

trimmed_eia = eia_df[trimmed_eia_columns]

trimmed_eia.columns = ['id', 'name', 'state', 'net_generation_mwh']

california_2016 = trimmed_eia[trimmed_eia.state == 'CA']

ranker_2016 = california_2016.groupby(["id", "name"]).net_generation_mwh.sum().reset_index()

la_paloma = trimmed_latimes_df[trimmed_latimes_df.plant_id == 55151]

trimmed_la_paloma = la_paloma.drop(['plant_id', 'plant_name_gen'], axis=1)

la_paloma_2016_gen = ranker_2016.set_index("id").at[55151, 'net_generation_mwh']

la_paloma_2016_cap = trimmed_la_paloma.set_index("year").at[2015.0, 'capacity_mwh']

la_paloma_2016_util = la_paloma_2016_gen / la_paloma_2016_cap

la_paloma_2016_df = pd.DataFrame([[
    2016.0,
    la_paloma_2016_gen,
    la_paloma_2016_cap,
    la_paloma_2016_util
]], columns=trimmed_la_paloma.columns)

combined_la_paloma = trimmed_la_paloma.append(la_paloma_2016_df)

combined_la_paloma.tail(5)

combined_la_paloma.tail(5).capacity_utilization.plot.bar()

ranker_2016.sort_values("net_generation_mwh", ascending=False).head(11).tail(10)

