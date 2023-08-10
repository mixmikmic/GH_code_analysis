from sleeplock import sleep
import dask

# A simple function to increment an integer...slowly!
def slow_inc(x):
    sleep(1)
    return x + 1

# A simple function to decrement an integer...slowly!
def slow_dec(x):
    sleep(1)
    return x - 1

get_ipython().run_line_magic('time', 'i_2 = slow_inc(2)')
i_2

get_ipython().run_line_magic('time', 'd_i_2 = slow_dec(i_2)')
d_i_2

delayed_inc = dask.delayed(slow_inc)
delayed_dec = dask.delayed(slow_dec)

get_ipython().run_line_magic('time', 'delayed_i_2 = delayed_inc(2)')
delayed_i_2

get_ipython().run_line_magic('time', 'delayed_d_i_2 = delayed_dec(delayed_i_2)')
delayed_d_i_2

get_ipython().run_line_magic('time', 'delayed_i_2.compute()')

get_ipython().run_line_magic('time', 'delayed_d_i_2.compute()')

get_ipython().run_line_magic('time', '_i_2, _d_i_2 = dask.compute(delayed_i_2, delayed_d_i_2)')
_i_2, _d_i_2

get_ipython().run_line_magic('time', 'persist_i_2 = delayed_inc(2).persist()')
persist_i_2

get_ipython().run_line_magic('time', 'persist_i_2.compute()')

get_ipython().run_line_magic('time', 'persist_d_i_2 = delayed_dec(persist_i_2)')
persist_d_i_2

get_ipython().run_line_magic('time', 'persist_d_i_2.compute()')

get_ipython().run_line_magic('time', '_i_2, _d_i_2 = dask.persist(delayed_i_2, delayed_d_i_2)')
_i_2, _d_i_2

delayed_i_2.key

persist_i_2.key

delayed_d_i_2.key

persist_d_i_2.key

# Short function to print out a Task Graph
def print_dask(dobj):
    for key in dobj.dask:
        print('{}:'.format(key))
        if isinstance(dobj.dask[key], tuple):
            print('    function:  {}'.format(dobj.dask[key][0]))
            print('    arguments: {}'.format(dobj.dask[key][1:]))
        else:
            print('    value: {}'.format(dobj.dask[key]))

print_dask(delayed_i_2)

print_dask(delayed_d_i_2)

delayed_i_2.visualize()

delayed_d_i_2.visualize()

persist_i_2.visualize()

persist_d_i_2.visualize()

