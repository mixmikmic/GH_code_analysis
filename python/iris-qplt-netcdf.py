import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import numpy.ma as ma

import iris
import iris.plot as iplt
import iris.quickplot as qplt



def main():
    # Load the "total electron content" cube.
    #filename = iris.sample_data_path('space_weather.nc')
    filename = 'sample_data/space_weather.nc'
    cube = iris.load_cube(filename, 'total electron content')

    # Explicitly mask negative electron content.
    cube.data = ma.masked_less(cube.data, 0)

    # Plot the cube using one hundred colour levels.
    qplt.contourf(cube, 100)
    plt.title('Total Electron Content')
    plt.xlabel('longitude / degrees')
    plt.ylabel('latitude / degrees')
    plt.gca().stock_img()
    plt.gca().coastlines()
    iplt.show()

main()



