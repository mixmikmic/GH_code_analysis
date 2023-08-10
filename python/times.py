import os, sys
sys.path.append(os.path.abspath('../../main/python'))

from thalesians.tsa.simulation import xtimes, times

for t in xtimes(0, 5): print(t)

xtimes(0, 5)

list(xtimes(0, 5))

times(0, 5)

list(range(0, 5))

ts = []
for t in xtimes(start=1):
    ts.append(t)
    if len(ts) == 5: break
ts

times(start=-3., stop=5., step=2.5)

import datetime as dt
times(dt.date(2017, 5, 5), dt.date(2017, 5, 10))

times(dt.time(8), dt.time(12), dt.timedelta(minutes=30))

times(dt.datetime(2017, 5, 10), dt.datetime(2017, 5, 5), dt.timedelta(days=-1))

import thalesians.tsa.random as rnd
times(0., 10., step=lambda x: rnd.exponential(2.5))

times(dt.datetime(2017, 5, 5, 8),
      dt.datetime(2017, 5, 5, 12),
      lambda x: rnd.exponential(dt.timedelta(minutes=30)))

