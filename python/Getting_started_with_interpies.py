import interpies

# show the plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

inFile = r'..\data\brtpgrd.gxf'
grid1 = interpies.open(inFile)

print('The cell size is {} m.'.format(grid1.cellsize))
print('The number of columns is {}.'.format(grid1.ncols))

grid1.info()

grid1.data

grid1.show()

ax = grid1.show(hs=False, cmap='coolwarm', cmap_norm='none', title='a grid in coolwarm and no hillshade')

# west, south, east, north
grid1.extent

grid2 = grid1.clip(xmin=450000, xmax=460000, ymin=4135000, ymax=4150000)
grid2.show();

grid1.clip(xmin=456000, xmax=466000, ymin=4120000, ymax=4130000).show();

# Horizontal derivative in the x (west-east) direction
# The vertical exaggeration (zf) has to be increased for the hillshade to look good.
ax = grid1.dx().show(zf=1000)

# tilt angle
ax = grid1.tilt().show()

ax = grid1.detrend().vi().hgm().show(title='Horizontal Gradient Magnitude of the Pseudogravity')

