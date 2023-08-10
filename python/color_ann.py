import pandas as pd
import numpy as np
import picture_stuff as pix
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Implement the function cosine_distance which computes a cosine-similarity-based distance between
# two numpy arrays.

def cosine_distance(x,y):
    #return x.dot(y)/(np.linalg.norm(x,axis=1)/np.linalg.norm(y))
    return x.dot(y)/(np.linalg.norm(x)/np.linalg.norm(y))

# Implement the function euclidean_distance which computes
# the Euclidean distance between two numpy arrays.

def euclidean_distance(x,y):
    #return np.sqrt(np.sum((x-y)**2,axis=1))
    return np.sqrt(np.sum((x-y)**2))

xx=np.array([0,3])
yy=np.array([0,1])
print xx
print yy
#cosine_distance(np.array([[0,1]]),np.array([[1,1]]))
print euclidean_distance(xx,yy)
print cosine_distance(xx,yy)

# Colors from http://www.peeron.com/cgi-bin/invcgis/colorguide.cgi
# and see also http://www.bartneck.de/2016/09/09/the-curious-case-of-lego-colors/
df_colors = pd.read_csv("peeron_colors.csv")

df_colors[df_colors['Color'].str.contains('red')]

X  = np.array(df_colors[['R','G','B']].astype('int'))
y  = np.array(df_colors['Color'])
ys = np.array(df_colors['LEGO No.'].astype('int'))

# Other color set:
curr_dir = '../../rebrickable data/2017-09/'
colors = pd.read_csv(curr_dir + 'colors.csv')

print "Rebrickable color file  {:8} rows".format(colors.shape[0])
print "Peerson color file      {:8} rows".format(len(y))

colors[colors['name'].str.contains('Red')]

def knn_dists(new_color,colors,distance=euclidean_distance,k=5):
    '''Calculates the k nearest neighbors within the group "colors"
       to a new value "new_color"
       INPUTS: new_color: numpy array of new data
               colors: numpy array of old data
               distance: distance function
               k: integer number of neighbors
       OUTPUT: numpy array of integers with ranked indices of neighbors
    '''
    distances = np.zeros(len(colors))  
    for idx in range(len(colors)):
        distances[idx]=distance(new_color,colors[idx])
    return np.argsort(distances)[:k]

test_idx = 3
#NOTE: idx=3: Brick yellow, idx=5: Light reddish violet, idx=23: Medium red

#closest = knn_dists(X[test_idx],X,distance=cosine_distance)
closest = knn_dists(X[test_idx],X)

print closest

X[test_idx]

result_colors = []
result_names = []
result_indices = []
for idx in closest:
    result_colors.append(X[idx])
    result_names.append(y[idx])
    result_indices.append(ys[idx])


print result_colors
print result_names
print result_indices

print X[test_idx]

picpath = "../../../brick_pics_mine/"
pic_list = ["3031-019.jpeg","3032-002.jpeg","3298-001.jpeg","3795-001.jpeg"]
#pic_list = ["3031-019.jpeg"]

def choose_region(image,ranges):
    '''selects subset of an image based on a list of x and y indices of the pixels
       INPUTS: image: numpy array
               ranges: list of lists [x_min,x_max,y_min,y_max]
       OUTPUTS: return_pics: list of numpy arrays
    '''
    return_pics = []
    for imrange in ranges:
        subset = image_filter(image,imrange)
        ave_color = average_color(subset)
        return_pics.append(subset)
    return return_pics

def image_filter(image,pix_range):
    '''returns the pixel from an image given min/max positions
       INPUTS: image: numpy array
               range: list [x_min,x_max,y_min,y_max]
       OUTPUTS: subset: numpy arrays
    '''
    x_min = pix_range[0]
    x_max = pix_range[1]
    y_min = pix_range[2]
    y_max = pix_range[3]
    print x_min,x_max,y_min,y_max
    subset = image[x_min:x_max,y_min:y_max]
    return subset

def set_ranges(x_size=299,y_size=299,
               x_from_center=.15,y_from_center=.15,
               x_span=10,y_span=10):
    '''Define 9 ranges within each range of pixels
       INPUTS: x_size=299,y_size=299: size of image
               x_from_center=.1,y_from_center=.1: fractional distance
                 from center for each off-center blob
               x_span=10,y_span=10: number of pixels in each range
       OUTPUTS: pix_ranges: list of lists [[x_min,xmax,y_min,y_max]]
    '''
    pix_ranges = []
    # Specify the lower left corner of each box
    left_x = int((x_size - x_span)/2)
    lower_y = int((y_size - y_span)/2)
    
    # Convert percentages from center to pixel steps
    step_x = int(x_size * x_from_center)
    step_y = int(y_size * y_from_center)

    for idx_x in range(-1,2):
        for idx_y in range(-1,2):
            pix_ranges.append([left_x + idx_x*step_x, left_x + idx_x*step_x + x_span,
                               lower_y + idx_y*step_y, lower_y + idx_y*step_y + y_span])
    return pix_ranges

def average_color(subset):
    '''returns average R,G,B values within subset
       INPUT: subset: numpy array with shape (m,n,3)
       OUTPUT numpy array with shape (3)
    '''
    return np.mean(subset,axis=(0,1))

fig, ax = plt.subplots(1,len(pic_list),figsize=(6,6))

for idx,pic in enumerate(pic_list):
    image_path = picpath + pic
    pic_pixels = pix.get_image(image_path).astype('float64')
    ax[idx].imshow(pic_pixels)

ranges = set_ranges()
image_path = picpath + pic_list[0]
pic_pixels = pix.get_image(image_path).astype('float64')
#pic_pixels = pic_pixels[:,:,::-1]
# NOTE: pic_pixels[:,:,::-1] is presented this way because of OpenCV's
# preference to treat RGB files as BGR, so they need to be reversed.

plt.imshow(pic_pixels)

# NOTE: regions is a list
regions = choose_region(pic_pixels,ranges)
regions = [region[:,:,::-1] for region in regions]


print [regions[idx].shape for idx in range(len(regions))]

idx=4
print average_color(regions[idx])
plt.imshow(regions[idx])

idx=4
closest = knn_dists(average_color(regions[idx]),X)

result_colors = []
result_names = []
result_indices = []
for idx in closest:
    result_colors.append(X[idx])
    result_names.append(y[idx])
    result_indices.append(ys[idx])

print closest
print result_colors
print result_names
print result_indices

