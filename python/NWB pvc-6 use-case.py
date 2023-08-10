from nixio import *
from utils.notebook import print_stats
from utils.plotting import Plotter
get_ipython().magic('matplotlib inline')

f = File.open("data/pvc-6.nix.h5", FileMode.ReadOnly)

print_stats(f.blocks)

block = f.blocks[0]

print_stats(block.data_arrays)
print_stats(block.multi_tags)

# find the data and sort it
time_series = [ts for ts in block.data_arrays if ts.type.endswith("time_series")]
time_series.sort(lambda a, b: cmp(a.name, b.name))
ts_count = len(time_series)

# plot the recorded signals
plotter = Plotter(lines=ts_count, width=1000, height=700)
for i in range(ts_count):
    plotter.add(time_series[i], subplot=i)
    
plotter.plot()

# get the tags
tags = list(block.multi_tags)

# plot the data
plotter = Plotter(lines=ts_count, width=1000, height=700)
for i, tag in enumerate(tags):
    plotter.add(tag.references[0], subplot=i)
    plotter.add(tag.positions, subplot=i, color="red", labels=tag.features[0].data)
        
plotter.plot()

f.close()



