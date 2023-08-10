import tensorflow as tf
import helpers as helpers
import matplotlib.pyplot as plt
import os
import numpy as np

data_sample=helpers.get_data("ECAL_rechit_occ_time_276582.hdf5",data_type='good')
print(data_sample.shape)
input_image=data_sample[0,:]
print(np.amax(input_image))
#print(input_image[0,:])
print(input_image.shape)

im=plt.imshow(input_image)
plt.show()
plt.clf()



import keras.models as models
trained_model=models.load_model(os.environ['BASEDIR']+"/models/autoencoder_v31_adam.h5")
input_image=np.reshape(input_image,(1,1,input_image.shape[0],input_image.shape[1]))
print(input_image.shape)

reconstructed_image=trained_model.predict(input_image)
print(reconstructed_image.shape)
reconstructed_image=np.reshape(reconstructed_image,(input_image.shape[2],input_image.shape[3]))
print(reconstructed_image.shape)
print(reconstructed_image[169,:])
im=plt.imshow(reconstructed_image)
plt.show()

