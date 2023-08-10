title = "Fetching data from a CSW catalog"
name = '2015-10-12-fetching_data'

get_ipython().magic('matplotlib inline')
import seaborn
seaborn.set(style='ticks')

import os
from datetime import datetime
from IPython.core.display import HTML

import warnings
warnings.simplefilter("ignore")

# Metadata and markdown generation.
hour = datetime.utcnow().strftime('%H:%M')
comments = "true"

date = '-'.join(name.split('-')[:3])
slug = '-'.join(name.split('-')[3:])

metadata = dict(title=title,
                date=date,
                hour=hour,
                comments=comments,
                slug=slug,
                name=name)

markdown = """Title: {title}
date:  {date} {hour}
comments: {comments}
slug: {slug}

{{% notebook {name}.ipynb cells[2:] %}}
""".format(**metadata)

content = os.path.abspath(os.path.join(os.getcwd(), os.pardir,
                                       os.pardir, '{}.md'.format(name)))

with open('{}'.format(content), 'w') as f:
    f.writelines(markdown)


html = """
<small>
<p> This post was written as an IPython notebook.  It is available for
<a href="http://ioos.github.com/system-test/downloads/
notebooks/%s.ipynb">download</a>.  You can also try an interactive version on
<a href="http://mybinder.org/repo/ioos/system-test/">binder</a>.</p>
<p></p>
""" % (name)

from datetime import datetime, timedelta

event_date = datetime(2015, 8, 15)

start = event_date - timedelta(days=4)
stop = event_date + timedelta(days=4)

spacing = 0.25

bbox = [-71.05-spacing, 42.28-spacing,
        -70.82+spacing, 42.38+spacing]

import iris
from utilities import CF_names

sos_name = 'sea_water_temperature'
name_list = CF_names[sos_name]

units = iris.unit.Unit('celsius')

from owslib import fes
from utilities import fes_date_filter

kw = dict(wildCard='*',
          escapeChar='\\',
          singleChar='?',
          propertyname='apiso:AnyText')

or_filt = fes.Or([fes.PropertyIsLike(literal=('*%s*' % val), **kw)
                  for val in name_list])

# Exclude ROMS Averages and History files.
not_filt = fes.Not([fes.PropertyIsLike(literal='*Averages*', **kw)])

begin, end = fes_date_filter(start, stop)
filter_list = [fes.And([fes.BBox(bbox), begin, end, or_filt, not_filt])]

from owslib.csw import CatalogueServiceWeb

csw = CatalogueServiceWeb('http://www.ngdc.noaa.gov/geoportal/csw',
                          timeout=60)

csw.getrecords2(constraints=filter_list, maxrecords=1000, esn='full')

fmt = '{:*^64}'.format
print(fmt(' Catalog information '))
print("CSW version: {}".format(csw.version))
print("Number of datasets available: {}".format(len(csw.records.keys())))

from utilities import service_urls

dap_urls = service_urls(csw.records, service='odp:url')
sos_urls = service_urls(csw.records, service='sos:url')

print(fmt(' SOS '))
for url in sos_urls:
    print('{}'.format(url))

print(fmt(' DAP '))
for url in dap_urls:
    print('{}.html'.format(url))

from utilities import is_station

non_stations = []
for url in dap_urls:
    try:
        if not is_station(url):
            non_stations.append(url)
    except RuntimeError as e:
        print("Could not access URL {}. {!r}".format(url, e))

dap_urls = non_stations

print(fmt(' Filtered DAP '))
for url in dap_urls:
    print('{}.html'.format(url))

from pyoos.collectors.ndbc.ndbc_sos import NdbcSos

collector_ndbc = NdbcSos()

collector_ndbc.set_bbox(bbox)
collector_ndbc.end_time = stop
collector_ndbc.start_time = start
collector_ndbc.variables = [sos_name]

ofrs = collector_ndbc.server.offerings
title = collector_ndbc.server.identification.title
print(fmt(' NDBC Collector offerings '))
print('{}: {} offerings'.format(title, len(ofrs)))

from utilities import collector2table, get_ndbc_longname

ndbc = collector2table(collector=collector_ndbc)

names = []
for s in ndbc['station']:
    try:
        name = get_ndbc_longname(s)
    except ValueError:
        name = s
    names.append(name)

ndbc['name'] = names

ndbc.set_index('name', inplace=True)
ndbc.head()

from pyoos.collectors.coops.coops_sos import CoopsSos

collector_coops = CoopsSos()

collector_coops.set_bbox(bbox)
collector_coops.end_time = stop
collector_coops.start_time = start
collector_coops.variables = [sos_name]

ofrs = collector_coops.server.offerings
title = collector_coops.server.identification.title
print(fmt(' Collector offerings '))
print('{}: {} offerings'.format(title, len(ofrs)))

from utilities import get_coops_metadata

coops = collector2table(collector=collector_coops)

names = []
for s in coops['station']:
    try:
        name = get_coops_metadata(s)[0]
    except ValueError:
        name = s
    names.append(name)

coops['name'] = names

coops.set_index('name', inplace=True)
coops.head()

from pandas import concat

all_obs = concat([coops, ndbc])

all_obs.head()

from pandas import DataFrame
from owslib.ows import ExceptionReport
from utilities import pyoos2df, save_timeseries

iris.FUTURE.netcdf_promote = True

data = dict()
col = 'sea_water_temperature (C)'
for station in all_obs.index:
    try:
        idx = all_obs['station'][station]
        df = pyoos2df(collector_ndbc, idx, df_name=station)
        if df.empty:
            df = pyoos2df(collector_coops, idx, df_name=station)
        data.update({idx: df[col]})
    except ExceptionReport as e:
        print("[{}] {}:\n{}".format(idx, station, e))

from pandas import date_range

index = date_range(start=start, end=stop, freq='1H')
for k, v in data.iteritems():
    data[k] = v.reindex(index=index, limit=1, method='nearest')

obs_data = DataFrame.from_dict(data)

obs_data.head()

import warnings
from iris.exceptions import (CoordinateNotFoundError, ConstraintMismatchError,
                             MergeError)
from utilities import (quick_load_cubes, proc_cube, is_model,
                       get_model_name, get_surface)

cubes = dict()
for k, url in enumerate(dap_urls):
    print('\n[Reading url {}/{}]: {}'.format(k+1, len(dap_urls), url))
    try:
        cube = quick_load_cubes(url, name_list,
                                callback=None, strict=True)
        if is_model(cube):
            cube = proc_cube(cube, bbox=bbox,
                             time=(start, stop), units=units)
        else:
            print("[Not model data]: {}".format(url))
            continue
        cube = get_surface(cube)
        mod_name, model_full_name = get_model_name(cube, url)
        cubes.update({mod_name: cube})
    except (RuntimeError, ValueError,
            ConstraintMismatchError, CoordinateNotFoundError,
            IndexError) as e:
        print('Cannot get cube for: {}\n{}'.format(url, e))

from iris.pandas import as_series
from utilities import (make_tree, get_nearest_water,
                       add_station, ensure_timeseries, remove_ssh)

model_data = dict()
for mod_name, cube in cubes.items():
    print(fmt(mod_name))
    try:
        tree, lon, lat = make_tree(cube)
    except CoordinateNotFoundError as e:
        print('Cannot make KDTree for: {}'.format(mod_name))
        continue
    # Get model series at observed locations.
    raw_series = dict()
    for station, obs in all_obs.iterrows():
        try:
            kw = dict(k=10, max_dist=0.08, min_var=0.01)
            args = cube, tree, obs.lon, obs.lat
            series, dist, idx = get_nearest_water(*args, **kw)
        except ValueError as e:
            status = "No Data"
            print('[{}] {}'.format(status, obs.name))
            continue
        if not series:
            status = "Land   "
        else:
            series = as_series(series)
            raw_series.update({obs['station']: series})
            status = "Water  "
        print('[{}] {}'.format(status, obs.name))
    if raw_series:  # Save that model series.
        model_data.update({mod_name: raw_series})
        del cube

import matplotlib.pyplot as plt

buoy = '44013'

fig , ax = plt.subplots(figsize=(11, 2.75))

obs_data[buoy].plot(ax=ax, label='Buoy')

for model in model_data.keys():
    try:
        model_data[model][buoy].plot(ax=ax, label=model)
    except KeyError:
        pass  # Could not find a model at this location.

leg = ax.legend()

buoy = '44029'

fig , ax = plt.subplots(figsize=(11, 2.75))

obs_data[buoy].plot(ax=ax, label='Buoy')

for model in model_data.keys():
    try:
        model_data[model][buoy].plot(ax=ax, label=model)
    except KeyError:
        pass  # Could not find a model at this location.

leg = ax.legend()

buoy = '8443970'

fig , ax = plt.subplots(figsize=(11, 2.75))

obs_data[buoy].plot(ax=ax, label='Buoy')

for model in model_data.keys():
    try:
        model_data[model][buoy].plot(ax=ax, label=model)
    except KeyError:
        pass  # Could not find a model at this location.

leg = ax.legend()

HTML(html)

