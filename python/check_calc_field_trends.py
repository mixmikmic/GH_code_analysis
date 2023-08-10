import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import iris.quickplot as qplt

get_ipython().magic('matplotlib inline')

cubes = iris.load('/g/data/r87/dbi599/temp/test_trend_field.nc')

print(cubes)

levels = [-2.5e9, -2.0e9, -1.5e9, -1.0e9, -0.5e9, 0, 0.5e9, 1.0e9, 1.5e9, 2.0e9, 2.5e9]

qplt.contourf(cubes[8], cmap='RdBu_r', levels=levels)
plt.gca().coastlines()
plt.show()

# 2D plots
iplt.plot(cubes[5])

2.5e9



