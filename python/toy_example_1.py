import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

block_size = 40

x = np.arange(0,block_size)
y = np.arange(0,block_size)
z = np.array([0])
order = 1

np.arange(10)

grid_matrix = np.meshgrid(np.arange(len(x)), 
                          np.arange(len(y)), 
                          np.arange(len(z)))

np.meshgrid(np.arange(1,4), np.arange(1,4), np.ones(3)) #, np.arange(1,1))

np.arange(1,4)

grid_matrix.shape
num_of_elements = grid_matrix.shape[0]

# construct element - reflects constructElements.m
def constructElement(x,y,z,order):
    grid

class Element(object):
    """Class definition (instead of Matlab stuct)"""
    
    def __init__(self, grid_matrix, x, y, z):
        self.Grid = grid_matrix
        self.Center = [x[grid_matrix[:,0]],
                       y[grid_matrix[:,1]],
                       z[grid_matrix[:,2]]]
        
        self.num_of_elements = grid_matrix.shape[0]
        
        self.Degree = np.empty(self.num_of_elements)
        self.Color = np.empty(self.num_of_elements)
        # fill with nan values
        self.Degree.fill(np.nan)
        self.Color.fill(np.nan)
        
    def find_neighbours(self):
        pass
        

E = Element(grid_matrix, x, y, np.zeros_like(x))

E.Center

grid_matrix[:,0]



