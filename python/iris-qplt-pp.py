import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import cartopy.crs as ccrs

import iris
import iris.plot as iplt
import iris.quickplot as qplt


def main():
    #fname = iris.sample_data_path('air_temp.pp')
    fname = 'sample_data/air_temp.pp'
    temperature = iris.load_cube(fname)

    # Plot #1: contourf with axes longitude from -180 to 180
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(121)
    qplt.contourf(temperature, 15)
    plt.gca().coastlines()

    # Plot #2: contourf with axes longitude from 0 to 360
    proj = ccrs.PlateCarree(central_longitude=-180.0)
    ax = plt.subplot(122, projection=proj)
    qplt.contourf(temperature, 15)
    plt.gca().coastlines()
    iplt.show()

main()

#fname = iris.sample_data_path('air_temp.pp')
fname = 'sample_data/air_temp.pp'
t0 = iris.load_cube(fname)

t0.xml()

whos



