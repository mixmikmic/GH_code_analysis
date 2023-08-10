from cablab import Cube
get_ipython().magic('matplotlib inline')

cube = Cube.open("C:\\Users\\Norman\\EOData\\ESDC\\low-res")

cube.data.variable_names

ds = cube.data.dataset()

ds

lst = ds['land_surface_temperature']

lst

lst_point = lst.sel(time='2006-06-15', lat=53, lon=11, method='nearest')

lst_point

lst.sel(lat=53, lon=11, method='nearest').plot()

lst.sel(time='2006-06-15', method='nearest').plot()

lst.sel(lon=11, method='nearest').plot()

lst.sel(lat=53, method='nearest').plot()

oz = ds['ozone']

oz

oz.sel(lat=53, lon=11, method='nearest').plot()

oz.sel(time='2006-06-15', method='nearest').plot()

oz.sel(lon=11, method='nearest').plot()

oz.sel(lat=53, method='nearest').plot()



