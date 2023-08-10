import app_stuff.src.picture_stuff as pix
import numpy as np
from PIL import Image, ImageOps
import os.path

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

image_dims = 299
scale_vals = (image_dims,image_dims)
border_percent = .3

camera = pix.initialize_camera()

keep_shooting = True
while keep_shooting:

    # Input a file name = brick shape: e.g. 3021
    pic_label = raw_input('Type label (file name):')
    extension = 1
    same_brick = True
    
    while same_brick:
        extension, filename = pix.increment_filename(pic_label,extension)

        # Shoot picture, crop and scale
        pic = pix.shoot_pic(camera)
        im = Image.fromarray(pic)
        newpic = ImageOps.fit(im, scale_vals, Image.ANTIALIAS,
                              border_percent, (.5,.5))
        np_newpic = np.array(newpic)
        fig, ax = plt.subplots(1,figsize=(8,8))
        ax.imshow(np_newpic[:,:,::-1], cmap=plt.cm.gray_r, interpolation="nearest")
        ax.grid(False)
        plt.show();
        
        next_action = raw_input(
            'Enter 0:reshoot, 1:next brick, q:quit, any other:save and shoot another')
        if next_action in ['q','Q','quit']:
            keep_shooting = False
            same_brick = False
        if next_action not in ['0','1','q','Q','quit']:
            pix.save_pic(filename,np_newpic)
            extension +=1    
        if next_action == '1':
            same_brick = False

# At the end of it all, release the camera
del(camera)

