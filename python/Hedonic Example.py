import os
import pandas as pd
import numpy as np
import orca
from urbansim.models import RegressionModel
from urbansim.utils import misc

orca.add_injectable("store", pd.HDFStore(os.path.join(misc.data_dir(), "sanfran_public.h5"), mode="r"))

@orca.table('buildings')
def buildings(store):
    df = store['buildings']
    return df

@orca.table('zones')
def zones(store):
    df = store['zones']
    return df

@orca.table('households')
def households(store):
    df = store['households']
    return df

@orca.table('parcels')
def parcels(store):
    df = store['parcels']
    return df

orca.broadcast('zones', 'buildings', cast_index=True, onto_on='zone_id')

@orca.column('households', 'income_quartile', cache=True)
def income_quartile(households):
    return pd.Series(pd.qcut(households.income, 4).labels,
                     index=households.index)

@orca.column('households', 'zone_id', cache=True)
def zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id)

@orca.column('zones', 'ave_unit_sqft')
def ave_unit_sqft(buildings, zones):
    s = buildings.unit_sqft[buildings.general_type == "Residential"]        .groupby(buildings.zone_id).quantile().apply(np.log1p)
    return s.reindex(zones.index).fillna(s.quantile())

@orca.column('zones', 'ave_lot_sqft')
def ave_lot_sqft(buildings, zones):
    s = buildings.unit_lot_size.groupby(buildings.zone_id).quantile().apply(np.log1p)
    return s.reindex(zones.index).fillna(s.quantile())

@orca.column('zones', 'sum_residential_units')
def sum_residential_units(buildings):
    return buildings.residential_units.groupby(buildings.zone_id).sum().apply(np.log1p)

@orca.column('zones', 'ave_income')
def ave_income(households, zones):
    s = households.income.groupby(households.zone_id).quantile().apply(np.log1p)
    return s.reindex(zones.index).fillna(s.quantile())

orca.add_injectable("building_type_map", {
    1: "Residential",
    2: "Residential",
    3: "Residential",
    4: "Office",
    5: "Hotel",
    6: "School",
    7: "Industrial",
    8: "Industrial",
    9: "Industrial",
    10: "Retail",
    11: "Retail",
    12: "Residential",
    13: "Retail",
    14: "Office"
})

@orca.column('buildings', 'zone_id', cache=True)
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)

@orca.column('buildings', 'general_type', cache=True)
def general_type(buildings, building_type_map):
    return buildings.building_type_id.map(building_type_map)

@orca.column('buildings', 'unit_sqft', cache=True)
def unit_sqft(buildings):
    return buildings.building_sqft / buildings.residential_units.replace(0, 1)

@orca.column('buildings', 'unit_lot_size', cache=True)
def unit_lot_size(buildings, parcels):
    return misc.reindex(parcels.parcel_size, buildings.parcel_id) /         buildings.residential_units.replace(0, 1)
    
@orca.column('parcels', 'parcel_size', cache=True)
def parcel_size(parcels):
    return parcels.shape_area * 10.764

rm = RegressionModel(
    fit_filters=[
        'unit_lot_size > 0',
        'year_built > 1000',
        'year_built < 2020',
        'unit_sqft > 100',
        'unit_sqft < 20000'
    ],
    predict_filters=[
        "general_type == 'Residential'"
    ],
    model_expression='np.log1p(residential_sales_price) ~ I(year_built < 1940)'
        '+ I(year_built > 2005) + np.log1p(unit_sqft) + np.log1p(unit_lot_size)'
        '+ sum_residential_units + ave_lot_sqft + ave_unit_sqft + ave_income',
    ytransform = np.exp
)     

merged_df = orca.merge_tables(target="buildings", tables=["buildings", "zones"], columns=rm.columns_used()) 

import utils
merged_df["year_built"] = merged_df.year_built.fillna(merged_df.year_built.quantile())
merged_df["residential_sales_price"] = merged_df.residential_sales_price.fillna(0)
merged_df["general_type"] = merged_df.general_type.fillna(merged_df.general_type.value_counts().idxmax())
_ = utils.deal_with_nas(merged_df)

rm.fit(merged_df).summary()

rm.predict(merged_df).describe()



