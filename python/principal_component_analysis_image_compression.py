import numpy as np
from matplotlib import pylab as plt

from PIL import Image
img = Image.open('shakira.jpg').convert('LA') #Convert the image to gray-scale.
img.save('shakira_gray.png') #Save gray-scale image.

def pca(img, num_components = 0):
    std_img = (img-np.mean(img.T,axis=1)).T # Make image 0 mean.
    [eig_vals, eig_vecs] = np.linalg.eig(np.cov(std_img)) # Find eigen values and eigen vectors of the covariance matrix.
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))] #Get the value-vector pair.
    eig_pairs.sort(key=lambda x: x[0], reverse=True) #Arrange in decreasing order of values, higher value is more information.
    # print('Eigenvalues in descending order:')
    # for i in eig_pairs:
    #     print(i[0])
    if num_components < np.size(eig_vecs, axis=1) and num_components >= 0: 
        eig_vecs = eig_vecs[:,range(num_components)] #Choose only required components, lesser components = lesser memory used
    #     print(eig_vecs)
    ax_img = np.dot(eig_vecs.T,std_img) #Create image on the new components.
    print('Shape of image is '+ str(img.shape))
    print('Shape of eigen vector is '+ str(eig_vecs.shape))
    print('Shape of compressed image is '+ str(ax_img.shape))
    print('Compression is '+str((ax_img.shape[0]*ax_img.shape[1] + eig_vecs.shape[0]*eig_vecs.shape[1])* 1.0 /(img.shape[0] * img.shape[1])))
    return eig_vecs, ax_img, eig_vals

from cmath import polar
def pca_driver():
    img = plt.imread('shakira_gray.png') # load an image
    print(img.shape)
    img = img[:, :, 1]
    plt.imshow(img)
    plt.gray()
    plt.show()
    print(img.shape)
    full_pc = np.size(img, axis=1)
    i = 1
    dist = []
    for num_components in range(0,full_pc+10,10):
        eig_vec, ax_img, eig_vals = pca(img, num_components)
        img_reconstructed = np.dot(eig_vec, ax_img).T+np.mean(img,axis=0)
        dist.append(np.linalg.norm(img-img_reconstructed,'fro'))
        #ax = plt.subplot(6,2,i,frame_on=False)
        #ax.xaxis.set_major_locator(plt.NullLocator())
        #ax.yaxis.set_major_locator(plt.NullLocator())
        i += 1
        plt.imshow(np.abs(img_reconstructed))
        plt.title('PCs # '+str(num_components))
        plt.gray()
        plt.show()

    plt.figure()
    plt.imshow(img)
    plt.title('All Components')
    plt.gray()
    plt.show()
    dist = dist/max(dist)
    plt.figure()
    plt.plot(range(0,full_pc+10,10),dist,'r')
    plt.axis([0,full_pc,0,1.1])
    plt.show()
    return

pca_driver()



