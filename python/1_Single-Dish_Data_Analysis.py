get_ipython().magic('pylab inline')



n = arange(10)

n

n[0]

n[-1]

n[2:3]

n[2:5]

n[:5]

n[5:]

t = arange(100)
x = cos(2 * pi * t / 40)
plot(t, x)

pl

get_ipython().magic('pinfo plot')

data = np.load('single_dish_tutorial.npz')

data.files

x = randn(500)
plot(x)

power = data['nd_power']
plot(power)





az = data['scan_az']
power = data['scan_power']
plot(az, power)
xlabel('Azimuth angle (degrees)')
ylabel('Power (counts)')











ra = data['raster_ra']
dec = data['raster_dec']
power = data['raster_power']
plot(ra, dec, '.')
axis('image')
xlabel('Right Ascension (degrees)')
ylabel('Declination (degrees)')

plot(power)
xlabel('Sample index')
ylabel('Flux density (Jy)')

pixel_size = 0.05
# Use the full (ra, dec) range for the plots...
grid_ra = np.arange(ra.min(), ra.max(), pixel_size)
grid_dec = np.arange(dec.min(), dec.max(), pixel_size)
# Or pick your own range to zoom in on the plots
#grid_ra = np.arange(200, 203, pixel_size)
#grid_dec = np.arange(-46, -41, pixel_size)
grid_power = griddata(ra, dec, power, grid_ra, grid_dec, interp='linear')

contour(grid_ra, grid_dec, grid_power, 50, cmap=cm.jet)
axis('image')
xlabel('Right Ascension (degrees)')
ylabel('Declination (degrees)')

extent=[grid_ra.min(), grid_ra.max(), grid_dec.min(), grid_dec.max()]
imshow(grid_power, extent=extent, origin='lower', cmap=cm.gray)
xlabel('Right Ascension (degrees)')
ylabel('Declination (degrees)')



