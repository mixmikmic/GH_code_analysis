import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
import pyoorb as oo
import ProtoMakeObs as pmo

orbits = pmo.readOrbits('pha20141031.des')

orbits = orbits[0:10]





