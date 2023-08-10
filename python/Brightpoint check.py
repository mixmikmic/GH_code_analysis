get_ipython().magic('matplotlib inline')

from ndreg import *
import matplotlib
import ndio.remote.neurodata as neurodata

import clarityviz as cl
import numpy as np
import nibabel as nib

reload(cl)

path = 's275_to_ara3_regis.nii'

im = nib.load(path)
im = im.get_data()
print(im.shape)

im_slice = im[:,:,660]

plt.imshow(im_slice, cmap='gray')
plt.show()

im_slice = im[600,201:300,201:400]

plt.imshow(im_slice, cmap='gray')
plt.show()

im_slice = im[600,201:300,201:400]

plt.imshow(im_slice, cmap='gray')
plt.show()

points = extract(im_slice, b_percentile = 0.9)

print(im_slice.shape)

print(points)

hist,bins = np.histogram(im_slice.flatten())

print(max(bins))

bim = np.zeros(im_slice.shape)

for point in points:
    bim[point[0], point[1]] = point[2]

plt.imshow(bim, cmap='gray')
plt.show()

plt.imshow(im_slice, cmap='gray')
plt.show()



def extract_bright_points(im, num_points=10000, b_percentile=0.75, optimize=True):
    """
    Function to extract points data from a np array representing a brain (i.e. an object loaded
    from a .nii file).
    :param im: The image array.
    :param num_points: The desired number of points to be downsampled to.
    :param b_percentile: The brightness percentile.
    :param optimize:
    :return: The bright points in a np array.
    """
    # obtaining threshold
    (values, bins) = np.histogram(im, bins=1000)
    cumValues = np.cumsum(values).astype(float)
    cumValues = (cumValues - cumValues.min()) / cumValues.ptp()

    maxIndex = np.argmax(cumValues > b_percentile) - 1
    threshold = bins[maxIndex]
    print(threshold)

    total = im.shape[0] * im.shape[1] * im.shape[2]
    #     print("Coverting to points...\ntoken=%s\ntotal=%d\nmax=%f\nthreshold=%f\nnum_points=%d" \
    #           %(self._token,total,self._max,threshold,num_points))
    print("(This will take couple minutes)")
    # threshold
    im_max = np.max(im)
    filt = im > threshold
    # a is just a container to hold another value for ValueError: too many values to unpack
    # x, y, z, a = np.where(filt)
    t = np.where(filt)
    x = t[0]
    y = t[1]
    z = t[2]
    v = im[filt]
    #     if optimize:
    #         self.discardImg()
    #     v = np.int16(255 * (np.float32(v) / np.float32(self._max)))
    l = v.shape
    print("Above threshold=%d" % (l))
    # sample

    total_points = l[0]
    print('total points: %d' % total_points)

    if not 0 <= num_points <= total_points:
        raise ValueError("Number of points given should be at most equal to total points: %d" % total_points)
    fraction = num_points / float(total_points)

    if fraction < 1.0:
        # np.random.random returns random floats in the half-open interval [0.0, 1.0)
        filt = np.random.random(size=l) < fraction
        print('v.shape:')
        print(l)
        #         print('x.size before downsample: %d' % x.size)
        #         print('y.size before downsample: %d' % y.size)
        #         print('z.size before downsample: %d' % z.size)
        print('v.size before downsample: %d' % v.size)
        x = x[filt]
        y = y[filt]
        z = z[filt]
        v = v[filt]
        #         print('x.size after downsample: %d' % x.size)
        #         print('y.size after downsample: %d' % y.size)
        #         print('z.size after downsample: %d' % z.size)
        print('v.size after downsample: %d' % v.size)
    points = np.vstack([x, y, z, v])
    points = np.transpose(points)
    print("Output num points: %d" % (points.shape[0]))
    print("Finished")
    return points

def extract(im, num_points=10000, b_percentile=0.75, optimize=True):
    """
    Function to extract points data from a np array representing a brain (i.e. an object loaded
    from a .nii file).
    :param im: The image array.
    :param num_points: The desired number of points to be downsampled to.
    :param b_percentile: The brightness percentile.
    :param optimize:
    :return: The bright points in a np array.
    """
    # obtaining threshold
    (values, bins) = np.histogram(im, bins=1000)
    cumValues = np.cumsum(values).astype(float)
    cumValues = (cumValues - cumValues.min()) / cumValues.ptp()

    maxIndex = np.argmax(cumValues > b_percentile) - 1
    threshold = bins[maxIndex]
    print(threshold)

#     total = im.shape[0] * im.shape[1] * im.shape[2]
    total = im.shape[0] * im.shape[1]

    #     print("Coverting to points...\ntoken=%s\ntotal=%d\nmax=%f\nthreshold=%f\nnum_points=%d" \
    #           %(self._token,total,self._max,threshold,num_points))
    print("(This will take couple minutes)")
    # threshold
    im_max = np.max(im)
    filt = im > threshold
    # a is just a container to hold another value for ValueError: too many values to unpack
    # x, y, z, a = np.where(filt)
    t = np.where(filt)
    x = t[0]
    y = t[1]
#     z = t[2]
    v = im[filt]
    #     if optimize:
    #         self.discardImg()
    #     v = np.int16(255 * (np.float32(v) / np.float32(self._max)))
    l = v.shape
    print("Above threshold=%d" % (l))
    # sample

    total_points = l[0]
    print('total points: %d' % total_points)

#     if not 0 <= num_points <= total_points:
#         raise ValueError("Number of points given should be at most equal to total points: %d" % total_points)
#     fraction = num_points / float(total_points)

#     if fraction < 1.0:
#         # np.random.random returns random floats in the half-open interval [0.0, 1.0)
#         filt = np.random.random(size=l) < fraction
#         print('v.shape:')
#         print(l)
#         #         print('x.size before downsample: %d' % x.size)
#         #         print('y.size before downsample: %d' % y.size)
#         #         print('z.size before downsample: %d' % z.size)
#         print('v.size before downsample: %d' % v.size)
#         x = x[filt]
#         y = y[filt]
#         z = z[filt]
#         v = v[filt]
#         #         print('x.size after downsample: %d' % x.size)
#         #         print('y.size after downsample: %d' % y.size)
#         #         print('z.size after downsample: %d' % z.size)
#         print('v.size after downsample: %d' % v.size)
    points = np.vstack([x, y, v])
#     points = np.vstack([x, y, z, v])
    points = np.transpose(points)
    print("Output num points: %d" % (points.shape[0]))
    print("Finished")
    return points

def plot_hist(im, title=''):
    hist,bins = np.histogram(im.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(im.flatten(),256,[0,256], color = 'r')
    plt.title(title)
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()



fig = plt.figure()
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(im[:,:,1000])
a.set_title('Before')
plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation ='horizontal')
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(im[600,:,:])
imgplot.set_clim(0.0,0.7)
a.set_title('After')
plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation='horizontal')

