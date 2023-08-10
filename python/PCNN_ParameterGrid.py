import sys, os
import time
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib
plt.ion()
from IPython import display
sys.path.insert(0, os.path.abspath('..'))
from pcnn import pcnn

# define the path to your image
img_name = os.path.join('/'.join(os.getcwd().split('/')[:-1]) ,'img','Lena.png')
# define the pcnn's parameters here
epochs = 20
# defines the linking strength
beta = 1.0
# defines the size of the receptive field
k_size = 3
kernel = np.ones((k_size,k_size))
# setting the center of the 3x3 kernel to zero, so the central neuron does not influence the next pulse
# kernel[1,1] = 0
# normalize the brightness using a ratio of gaussians approach
do_rog = False
# define factors to dcrease image size. Resulting image will be h/a x w/b if (a,b) is given
scales = (2,2)

gen_list = []
# choose three values for V here
V_list = [0.3,0.5,1]
# choose three values for alpha here
alpha_list = [0.05,0.2,0.3]
# choose the delay after each plot here
pause_val = 0.4
for V in V_list:
    for alpha in alpha_list:
        p = pcnn(kernel=kernel, epochs=epochs, V=V, alpha=alpha, beta=beta,
                 do_rog=do_rog, scales=scales)
        gen_list.append(p.get_gen(img_name))

f, axarr = plt.subplots(3, 3, figsize=(20,20))
f.patch.set_facecolor('#bbbbbb')
f.tight_layout()
axis_list = []

g = 0
# init plot
for i in range(3):
    for j in range(3):
        img = next(gen_list[g])
        im_plot = axarr[i,j].imshow(img, 'gray', vmin=0, vmax=1)
        axarr[i,j].axis('off')
        axarr[i,j].set_title("V={}; alpha={}".format(V_list[i],alpha_list[j]))
        axis_list.append(im_plot)
        g = g+1

plt.show()
# plt.pause(0.5)
display.clear_output(wait=True)

for epoch in range(1,epochs):
    for i, gen in enumerate(gen_list):
        img = next(gen)
        axis_list[i].set_data(img)
    plt.pause(pause_val)
    plt.show()
    display.display(f)
    print("Epoch {}".format(epoch))
    display.clear_output(wait=True)

