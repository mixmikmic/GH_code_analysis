# Import all methods
import json 
data_file = open('../data/workfile.json')
data = json.load(data_file)

print(data)

# Import a specific method
from os import path
print path.basename('../data/workfile.json')

# Alias modules
import numpy
print numpy.__version__
import numpy as np # Alias module
print np.__version__

