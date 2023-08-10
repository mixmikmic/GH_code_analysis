import time
import dask
import dask.multiprocessing

@dask.delayed
def add(x, y):
    time.sleep((x + y)/10.0)
    return x + y

@dask.delayed
def inc(x):
    time.sleep(x/10.0)
    return x + 1

@dask.delayed
def dbl(x):
    time.sleep(x/10.0)
    return 2*x

@dask.delayed
def dsum(*args):
    s = sum(*args)
    time.sleep(s/10.0)
    return s

data = [1,3,2,0]

sum_odds = dsum(inc(dbl(x)) for x in data)
sum_odds

sum_odds.visualize()

get_ipython().run_line_magic('prun', 'sum_odds.compute(get=dask.threaded.get, num_workers=4)')

get_ipython().run_line_magic('prun', 'sum_odds.compute(get=dask.get)')

from dask.diagnostics import Profiler, visualize
from bokeh.io import output_notebook
output_notebook()

with Profiler() as p:
    sum_odds.compute(get=dask.threaded.get, num_workers=4)
    
visualize(p)

from dask.diagnostics import ProgressBar

with ProgressBar():
    sum_odds.compute(get=dask.threaded.get, num_workers=4)



