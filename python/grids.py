get_ipython().system('pip install -q ipythonblocks')

from ipythonblocks import BlockGrid

help(BlockGrid)

grid = BlockGrid(10, 10, fill=(123, 234, 123))

grid

grid[0,0]

grid[0,0].red = 100
grid[0,0].green = 15
grid[0,0].blue = 15

grid[0,0]

