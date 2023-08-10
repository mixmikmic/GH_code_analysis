import os
import cpi
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)

years = range(1990, 2016)

whitelist = pd.DataFrame([
    ('10', 'Total, all industries', 'total'),
    ('111', 'Crop production', 'crops'),
    ('1151', 'Support activities for crop production', 'crops'),
], columns=['industry_code', 'industry_name', 'industry_group'])

path_template = './data/{}.annual.singlefile.csv'

area_titles = pd.read_csv("./data/area_titles.csv")

for year in years:
    print "Transforming {}".format(year)
    
    # Read in the csv
    df = pd.read_csv(path_template.format(year), dtype={"area_fips": str})
    
    # Decode the area titles
    df = df.merge(area_titles, on="area_fips", how="inner")

    # Filter it down to desired industries using whitelist
    filtered_df = df.merge(whitelist, on='industry_code', how="inner")
    
    # Filter it down to the statewide aggregation level for each industry
    state_df = filtered_df[
        # Statewide totals for all industries
        ((filtered_df.agglvl_code == 50) & (filtered_df.industry_group == 'total')) |
        # Statewide totals for our selected industries
        (
            (filtered_df.agglvl_code.isin([55, 56])) &
            (filtered_df.own_code == 5) &
            (filtered_df.industry_group == 'crops')
        )
    ]
    
    # Filter it down to the county aggregation level for each industry
    county_df = filtered_df[
        # County totals for all industries
        ((filtered_df.agglvl_code == 70) & (filtered_df.industry_group == 'total')) |
        # County totals for our selected industries
        (
            (filtered_df.agglvl_code.isin([75, 76])) &
            (filtered_df.own_code == 5) &
            (filtered_df.industry_group == 'crops')
        )
    ]

    # Trim to only the columns we want
    trimmed_columns = [
        'area_fips',
        'area_title',
        'industry_code',
        'industry_name',
        'industry_group',
        'agglvl_code',
        'year',
        'own_code',
        'avg_annual_pay',
        'annual_avg_emplvl',
        'total_annual_wages',
    ]
    trimmed_state_df = state_df[trimmed_columns]
    trimmed_county_df = county_df[trimmed_columns]
    
    # Adjust wages for inflation
    trimmed_state_df['total_annual_wages_2015'] = trimmed_state_df.apply(
        lambda x: cpi.to_2015_dollars(x.total_annual_wages, x.year),
        axis=1
    )
    trimmed_county_df['total_annual_wages_2015'] = trimmed_county_df.apply(
        lambda x: cpi.to_2015_dollars(x.total_annual_wages, x.year),
        axis=1
    )
    
    # Group totals by industry group
    groupby = [
        'year',
        'area_fips',
        'area_title',
        'industry_group'
    ]
    aggregation = {
        'annual_avg_emplvl': 'sum',
        'total_annual_wages_2015': 'sum'
    }
    grouped_state_df = trimmed_state_df.groupby(groupby).agg(aggregation).reset_index()
    grouped_county_df = trimmed_county_df.groupby(groupby).agg(aggregation).reset_index()
    
    # Recalculate average pay for the new group
    grouped_state_df['avg_annual_pay_2015'] = (
        grouped_state_df.total_annual_wages_2015 / grouped_state_df.annual_avg_emplvl
    )
    grouped_county_df['avg_annual_pay_2015'] = (
        grouped_county_df.total_annual_wages_2015 / grouped_county_df.annual_avg_emplvl
    )
    
    # Write out each annual file separately
    grouped_state_df.to_csv("./data/transformed_state_{}.csv".format(year), index=False)
    grouped_county_df.to_csv("./data/transformed_county_{}.csv".format(year), index=False)

combined_state_df = pd.concat(
    [pd.read_csv("./data/transformed_state_{}.csv".format(year), dtype={"area_fips": str}) for year in years],
    ignore_index=True
)

combined_county_df = pd.concat(
    [pd.read_csv("./data/transformed_county_{}.csv".format(year), dtype={"area_fips": str}) for year in years],
    ignore_index=True
)

combined_state_df.to_csv("./data/transformed_state.csv", index=False)

combined_county_df.to_csv("./data/transformed_county.csv", index=False)

