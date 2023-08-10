import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('pylab', 'inline')
import PIL
import os
import glob
import csv
import h5py

Bag_Name = '/home/james/bagfiles/2017-08-29-12-34-32.bag'
Image_output_dir = '/home/james/bagfiles/Ros_images'
bridge = CvBridge()
bag = rosbag.Bag(Bag_Name)
bag.get_type_and_topic_info()

Image_topic = '/camera/image_raw'
Lights_topic ='/vehicle/traffic_lights'

# pull a sample image and plot from the bag
for topic, msg, t in bag.read_messages(Image_topic):
    cv_image_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
#     cv_image_bgr = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)
    
    print("Time:", t)
    break

# pull a sample data and plot from the bag
for topic, msg, t in bag.read_messages(Lights_topic):
    print(msg.lights[0].pose.pose)
    
    print("Time:", t)
    break
    
    
bag.close();

# Lets write some PNGs for inspection
bridge = CvBridge()
bag = rosbag.Bag(Bag_Name)
count = 0

for topic, msg, t in bag.read_messages(Image_topic):
    cv_image_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    cv2.imwrite(os.path.join(Image_output_dir, "frame%06i.png" % count), cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR))
#     print "Wrote image %i" % count
    count+=1
bag.close();

# lets load all the images into memory to save them in a giant array
bridge = CvBridge()
bag = rosbag.Bag(Bag_Name)
count = 0
IM_Shape = cv_image_rgb.shape

# Pre-allocate memory to fill
row_count = bag.get_message_count(Image_topic)
Images_RGB = np.empty((row_count,IM_Shape[0], IM_Shape[1], IM_Shape[2]),dtype=np.uint8)
ARR = np.arange(row_count)
Images_index = np.empty(row_count,dtype=np.uint16)

for topic, msg, t in bag.read_messages(Image_topic):
    cv_image_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    Images_RGB[count,:,:,:]  = cv_image_rgb
    Images_index[count] = count

    count+=1
bag.close();

# Load the image classes into memory
# Red_light:0 Yellow_light:1 Green_light:2 Dontuse: 3 no_light: 4
c = 0
row_count = bag.get_message_count(Image_topic)
image_class = np.empty(row_count+1)

with open('/home/james/bagfiles/image_classes.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        image_class[c] =int(row[1])
        c+=1

# View so random images or data
ind =np.random.randint(0,row_count)
ind =875
print(image_class[ind])
plt.imshow(Images_RGB[ind])

h5f = h5py.File('/home/james/bagfiles/Track_Images.h5', 'w')
data_entry = h5f.create_group('Training_Data')
data_entry.create_dataset('Images_RGB', data=Images_RGB)
data_entry.create_dataset('image_class', data=image_class)
h5f.close()













































