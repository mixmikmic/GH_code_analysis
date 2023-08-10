get_ipython().magic('matplotlib inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.cluster.vq import *

#Get data
original_image_array = mpimg.imread("_12800_15200.tif")
flat_array = original_image_array.flatten()


#Clustering and reshaping
k = 6
centroids, variance = kmeans(flat_array.astype(float), k)
map_px , distance = vq(flat_array, centroids)
cluster_image = map_px.reshape(original_image_array.shape[0],original_image_array.shape[1]
                              ,original_image_array.shape[2])


#Plot original image
fig = plt.figure(figsize=(12,6))
fig.suptitle("Clustering on original image ", fontsize= 16)
ax = plt.subplot(1,2,1)
plt.axis('off')
ax.set_title('Original Image')
plt.imshow(original_image_array)


#Plot cluster image
ax = plt.subplot(1,2,2)
plt.axis('off')
ax.set_title(str(k) + " Clusters") 
cluster_image_gray = cluster_image[:,:,0]
plt.imshow(cluster_image_gray)



