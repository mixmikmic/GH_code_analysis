station = 'ECO09113499'
year = 2017

get_ipython().run_line_magic('matplotlib', 'inline')

import papermill as pm

import bikes

data = bikes.get_velo_data(station, year)

first_day = '%i0101' % year
data[:first_day].North.plot()

data[:first_day].South.plot()

first = data[:'%i0110' % year]
hourly = first.resample('H').mean()
(hourly.North - hourly.South).plot()

average = data.resample('W').mean()
total = data.resample('W').sum()

average.plot()

total.plot()

pm.record("year", year)
pm.record("station", station)
pm.record("sum", dict(north=list(total.North),
                      south=list(total.North),
                      total=list(total.Total)))
pm.record("average", dict(north=list(average.North),
                          south=list(average.South),
                          total=list(average.Total)))
pm.record("index", list(average.index))

