import numpy as np
import pandas as pd

# the point cloud filename from semantic3D
filename_pts = "./data/train/bildstein_station3_xyz_intensity_rgb.txt"

# get each point one per line into a list if the line has 7 components: xyzirgb
xyzrgb = [pnt.strip().split(' ') for pnt in open(filename_pts, 'r') if len(pnt.strip().split(' ')) == 7]

# get the x, y, z and r, g, b components as floats of each point 
xyzrgbs = np.array([[float(value) for value in pnt] for pnt in xyzrgb])

# better to do all operations per line instead of looping through it twice
xyzrgbs = [[float(value) for value in pnt.strip().split(' ')] for pnt 
           in open(filename_pts, 'r') if len(pnt.strip().split(' ')) == 7]

np.shape(xyzrgbs)

# load everything into a pandas dataframe for easier handling
df = pd.DataFrame(xyzrgbs, columns=['x', 'y', 'z', 'i', 'red', 'green', 'blue'])

df.head()

df.to_csv("bildstein_station3.csv", index=False)

df[['x','y','z','red','green','blue']].to_csv("bildstein_station3.csv", index=False)



