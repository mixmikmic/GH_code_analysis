# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
get_ipython().magic('matplotlib inline')

# set display defaults
plt.rcParams['figure.figsize'] = (10,10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
# We added it to PYTHONPATH (e.g. from ~/.bash_profile)

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

test_images_file = 'Emotion6/em6_posneg_test.txt'
samples = {}

with open(test_images_file) as f:
    for line in f:
        columns = line.split()
        label = columns[1]
        if not label in samples:
            samples[label] = []
        samples[label].append(columns[0])

for key in samples.keys():
    print 'key \"%s\": %i samples' % (key, len(samples[key]))

# todo: save the final models to experiments/Emotion6 folder, create "deploy" models
model_definition = 'Emotion6/e2.deploy.prototxt'
model_weights    = 'Emotion6/e2.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(model_definition,
         model_weights,
         caffe.TEST)
print 'Net model sucessfully loaded'

# load the mean places image 
mu = np.load('Emotion6/places205CNN_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

def classify_image(image_file):
    image = caffe.io.load_image(image_file)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    # perform classification
    net.forward()
    
def show_image(image_file):
    image = caffe.io.load_image(image_file)
    plt.imshow(image)
    plt.show()

classification_results = []
for label in samples.keys():
    for image_file in samples[label]:
        classify_image(image_file)
        # obtain the output probabilities
        output_prob = net.blobs['prob'].data[0]
        classification_results.append([image_file,int(label),output_prob.argmax(),output_prob.max()])

labels = [label[1] for label in classification_results]
predictions = [label[2] for label in classification_results]
from sklearn.metrics import accuracy_score
acc = accuracy_score(labels, predictions)
print 'Testing: Accuracy is %f' % acc

true_positive_top5  = sorted([elem for elem in classification_results if elem[1]==elem[2]==1],key=lambda x:x[3])[-5::]
false_positive_top5 = sorted([elem for elem in classification_results if (elem[1]==0 and elem[2]==1)],key=lambda x:x[3])[-5::]
true_negative_top5  = sorted([elem for elem in classification_results if elem[1]==elem[2]==0],key=lambda x:x[3])[-5::]
false_negative_top5 = sorted([elem for elem in classification_results if (elem[1]==1 and elem[2]==0)],key=lambda x:x[3])[-5::]


def collate_images(files, output = 'collated.jpg'):
    import sys
    from PIL import Image
    images = map(Image.open, files)
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    new_im.save(output)

def collate_and_show(files, output='collated.jpg'):
    collate_images(files,output)
    plt.imshow(caffe.io.load_image(output))
    plt.show()

print 'True Positive top 5'
collate_and_show([elem[0] for elem in true_positive_top5],'tp_top5.jpg')
print 'False Positive top 5'
collate_and_show([elem[0] for elem in false_positive_top5],'fp_top5.jpg')
print 'True Negative top 5'
collate_and_show([elem[0] for elem in true_negative_top5],'tn_top5.jpg')
print 'False Negative top 5'
collate_and_show([elem[0] for elem in false_negative_top5],'fn_top5.jpg')
    

# first of all let's choose a particular image
test_image = '../databases/Emotion6/images/joy/13.jpg'
show_image(test_image)
classify_image(test_image)

output_prob = net.blobs['prob'].data[0]
print 'Classification result is %i with probability %f' %(output_prob.argmax(), output_prob.max())

# for each layer, show the output shape
print 'layer outputs:'
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
    
print 'layer parameters:'
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    print 'data shape: '
    print data.shape# normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')

# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))

vis_square(net.blobs['conv5'].data[0,:])

net.blobs['prob'].data[0]



