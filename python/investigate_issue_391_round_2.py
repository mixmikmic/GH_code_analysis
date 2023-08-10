# Change the CWD to be the root so that the giles_conf.json file is found in the correct location
import os
os.chdir("../../../")

import pandas as pd
pd.options.display.float_format = '{:.6f}'.format

import pandas as pd
import emission.core.get_database as edb
import bson.objectid as boi
import numpy as np
import scipy.interpolate as spi
import arrow

import emission.storage.decorations.analysis_timeseries_queries as esda
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.timeseries.builtin_timeseries as estb
import emission.core.wrapper.entry as ecwe
import emission.core.wrapper.location as ecwl
import emission.analysis.intake.cleaning.location_smoothing as eaicl

import emission.storage.timeseries.timequery as estt

import emission.analysis.plotting.geojson.geojson_feature_converter as gfc
import emission.analysis.plotting.leaflet_osm.ipython_helper as ipy
import emission.analysis.plotting.leaflet_osm.our_plotter as lo

edb.get_timeseries_db().distinct('user_id')

from uuid import UUID

curr_id = UUID('2e0fa90c-59fe-4471-973d-ee7e1f93ae00')
tq = estt.TimeQuery("data.ts", 1469065859.779821, 1469073905.999965)
locs_df = esda.get_data_df("background/filtered_location", curr_id, time_query=tq)

locs_df.shape

ipy.inline_maps([lo.get_maps_for_geojson_unsectioned([gfc.get_feature_list_from_df(locs_df)])])

locs_df[["fmt_time", "ts", "metadata_write_ts", "latitude", "longitude"]]

filled_locs_df = eaicl._ios_fill_fake_data(locs_df)

with_speeds_df = eaicl.add_dist_heading_speed(filled_locs_df)

with_speeds_df[53:][["fmt_time", "ts", "metadata_write_ts", "latitude", "longitude", "speed", "distance"]]

jumps = with_speeds_df[(with_speeds_df.speed > 2.6585250029) & (with_speeds_df.distance > 100)].index

jumps

for jump in jumps:
    print jump
    jump_to = with_speeds_df[(with_speeds_df.index < jump) & (with_speeds_df.distance > 100)].index[-1]
    print jump_to

x = [1,2,3]
x.insert(0,0)

