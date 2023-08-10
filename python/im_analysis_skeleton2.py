get_ipython().magic('matplotlib inline')

import Tkinter as Tk, tkFileDialog
import os, sys
import javabridge as jv
import bioformats as bf
import matplotlib
matplotlib.use("TkAgg") #without this code, Tkinter and Matplotlib don't play nicely
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

jv.start_vm(class_path=bf.JARS, max_heap_size='12G')

root = Tk.Tk()
root.withdraw() #hiding root alllows file diaglog GUI to be shown without any other GUI elements
file_full_path = tkFileDialog.askopenfilename()
filepath, filename = os.path.split(file_full_path)
os.chdir(os.path.dirname(file_full_path))

print('')
print('User Selected:  %s' %filename)
print('')

md = bf.get_omexml_metadata(file_full_path)
ome = bf.OMEXML(md)
iome = ome.image(0) # e.g. first image
#print(ome.image_count)

print('')
print('Image Name:  %s' %iome.get_Name())
print('Image ID:  %s' %iome.get_ID()) #what is image ID?
print('Acquisition Date:  %s'  %iome.AcquisitionDate)
print('')

print('Bit Depth:  %s' %iome.Pixels.get_PixelType())
print('XYZ Dimensions:  %s x %s x %s pixels' %(iome.Pixels.get_SizeX(),iome.Pixels.get_SizeY(),iome.Pixels.get_SizeZ()))
print('Time Points:  %s' %iome.Pixels.get_SizeT())
print('DimensionOrder:  %s' %iome.Pixels.DimensionOrder)
#print('get_DimensionOrder:  %s' %iome.Pixels.get_DimensionOrder()) #what is the difference between get_DimensionOrder() and DimensionOrder?
print('Channels:  %s' %iome.Pixels.get_SizeC())
print('Ch1:  %s' %iome.Pixels.Channel(0).Name)
print('Ch2:  %s' %iome.Pixels.Channel(1).Name)
print('')

reader = bf.ImageReader(file_full_path)

raw_data = []
for z in range(iome.Pixels.get_SizeZ()):
    raw_image = reader.read(z=z, series=0, rescale=False)
    raw_data.append(raw_image)
    
raw_data = np.array(raw_data)
print('Your array of image has the shape:  %s' %str(raw_data.shape))

plt.imshow(raw_data[16, :, :, 0], cmap=cm.gray)
plt.show()

jv.kill_vm()



