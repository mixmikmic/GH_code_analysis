from fauxtograph import VAE, GAN, VAEGAN, get_paths, image_resize
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

paths = get_paths('/home/ubuntu/Data/hubble/')

image_resize(paths, '/home/ubuntu/Data/hubble_resized/', 96, 96)

paths = get_paths('/home/ubuntu/Data/hubble_resized/')

vg = VAEGAN(img_width=96, img_height=96)
x_all = vg.load_images(paths)

m_path = '/home/ubuntu/Data/VAEGAN_training_model/'
im_path = '/home/ubuntu/Data/VAEGAN_training_images/'
vg.fit(x_all, save_freq=2, pic_freq=30, n_epochs=4, model_path = m_path, img_path=im_path, mirroring=True)

vg.save('/home/ubuntu/Data/VAEGAN_models/', 'test')

loader ={}
loader['enc'] = '/home/ubuntu/Data/VAEGAN_models/test_enc.h5'
loader['dec'] = '/home/ubuntu/Data/VAEGAN_models/test_dec.h5'
loader['disc'] = '/home/ubuntu/Data/VAEGAN_models/test_disc.h5'
loader['enc_opt'] = '/home/ubuntu/Data/VAEGAN_models/test_enc_opt.h5'
loader['dec_opt'] = '/home/ubuntu/Data/VAEGAN_models/test_dec_opt.h5'
loader['disc_opt'] = '/home/ubuntu/Data/VAEGAN_models/test_disc_opt.h5'
loader['meta'] = '/home/ubuntu/Data/VAEGAN_models/test_meta.json'
vg2 = VAEGAN.load(**loader)

import numpy as np
shape = 10, vg2.latent_width
random_data = np.random.standard_normal(shape).astype('f')*3.
images = vg2.inverse_transform(random_data, test=True)
plt.figure(figsize=(16,3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i])
    plt.axis("off")
plt.show()

