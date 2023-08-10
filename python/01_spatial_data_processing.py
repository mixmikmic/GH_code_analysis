# our convention is to alias PySAL, NumPy and Pandas
import pysal as ps
import numpy as np
import pandas as pd

# check the versions
ps.version

np.version.short_version

pd.__version__

get_ipython().system('head data/mexico.csv')

f = ps.open("data/mexico.csv")
vnames = ["pcgdp%d"%decade for decade in range(1940, 2010, 10)]

vnames

Y = np.transpose(np.array([f.by_col[v] for v in vnames]))

Y

state = f.by_col['State']

f.close() # done with the file

state

# us counties example

### First the attributes from the dbf

dbf = ps.open('data/NAT.dbf')
header = dbf.header

header

# Read all the numeric variables into a big array
# find the first offset we need
start_col = header.index("SOUTH")

start_col

vars = header[8:]

vars

nat_array = np.array([np.array(dbf.by_col(var)) for var in vars])

nat_array.shape

nat_array = nat_array.T

nat_array.shape

dbf.close() # done with the dbf file

shp_file = ps.open("data/NAT.shp")

shp_file.header

len(shp_file)

shapes = [ shp_file.next() for i in xrange(len(shp_file)) ]

type(shapes)

len(shapes)



type(shapes)

s0 = shapes[0]

s0

dir(s0)

shp_file.close()

import json

f = open('data/nat.json')

fj = json.load(f)

f.close()

fj.keys()

features = []
for feature in fj['features']:
    features.append(feature)

features[0]







