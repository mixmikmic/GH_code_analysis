import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

get_ipython().magic('matplotlib inline')
# this file should be run from {caffe_root}/examples (otherwise change this line)
caffe_root = '/media/hermetico/2TB/frameworks/caffe/'  

activities_net = {
    'model': 'models/finetuning-googlenet/reference_activitiesnet.caffemodel',
    'deploy': 'models/finetuning-googlenet/deploy.prototxt'
}
# the paths for the labels and for the dataset
labels_path = os.path.join(caffe_root, 'data/daily_activities/labels.txt')
test_dataset_path = os.path.join(caffe_root, 'data/daily_activities/test.txt')
sys.path.insert(0, caffe_root + 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# dataset stuff
test_dataset = [ line.split(' ') 
                for line in list(np.loadtxt(test_dataset_path, str, delimiter='\n'))
                if len(line.split(' ')) == 2 # there are some images with wierd paths
               ]
test_paths, test_labels = zip(*[(path, int(label)) for path, label in test_dataset])
labels = list(np.loadtxt(labels_path, str, delimiter='\n'))
NUM_LABELS = len(labels)
print "%i labels loaded" % NUM_LABELS
print "%i test images" % (len(test_paths))

# loads activities net 
activitiesnet = caffe.Classifier(activities_net['deploy'], activities_net['model'], caffe.TEST)
activitiesnet.blobs['data'].reshape(1,3,227,227)

# Preprocessing for caffe inputs
transformer = caffe.io.Transformer({'data': activitiesnet.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

from scipy import misc
import random

def disp_preds(net, labels, image_path):
    """This function recieves a net and an image_path, and returns the top 5 prediction"""
    plt.imshow(misc.imread(image_path))
    im = caffe.io.load_image(image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net.forward()
    layer = 'prob'
    if not layer in net.blobs:
        layer = 'probs'
    top_5 = np.sort(net.blobs[layer].data[0].flatten())[-1:-6:-1]
    top_5_index = net.blobs[layer].data[0].flatten().argsort()[-1:-6:-1]
    for i in range(len(top_5_index)):
        print "%.0f%%: %s"%( top_5[i] * 100., labels[top_5_index[i]])
        

def disp_pred(net, labels, image_path):
    """This function recieves a net and an image_path, and returns the top 5 prediction"""
    plt.imshow(misc.imread(image_path))
    im = caffe.io.load_image(image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net.forward()
    layer = 'prob'
    if not layer in net.blobs:
        layer = 'probs'
    top = np.sort(net.blobs[layer].data[0].flatten())[-1]
    top_index = net.blobs[layer].data[0].flatten().argsort()[-1]
    print "%.0f%%: %s"%( top * 100., labels[top_index])


index = random.randint(0, len(test_paths))
print "Label: ", labels[test_labels[index]]
print "\nBest prediction"
disp_pred(activitiesnet, labels,  test_paths[index])
print "\nBest 5 predictions"
disp_preds(activitiesnet, labels,  test_paths[index])

paths = [[] for _ in range(NUM_LABELS)]
# each index index paths will contain a list of  pictures associated to a category
for i, path in enumerate(test_paths):
    paths[test_labels[i]].append(path)

# number of images per category
print [len(examples) for examples in paths]

y_pos = np.arange(NUM_LABELS)
num_examples = [len(examples) for examples in paths]
plt.barh(y_pos, num_examples,  alpha=1)
plt.yticks(y_pos, labels)
plt.xlabel('Number of pictures')
plt.title('List of categories and number of pictures')
plt.show()

def eval_category(net, category, label, layer='probs'):
    correct = 0.
    for path in category:
        processed_image = im = caffe.io.load_image(path)
        net.blobs['data'].data[...] = transformer.preprocess('data', processed_image)
        out = net.forward()
        top_label = net.blobs[layer].data[0].flatten().argsort()[-1]
        if top_label == label:
            correct += 1
    return correct / len(category)
        
def eval_categories(net, categories):
    """evaluates the net"""
    accuracies = np.zeros(len(categories))
    for i, category in enumerate(categories):
        accuracies[i] = eval_category(net,category, i )
    return accuracies * 100

accuracies = eval_categories(activitiesnet, paths)

for i, acc in enumerate(accuracies):
    print "%s: %i" %(labels[i], acc)

y_pos = np.arange(NUM_LABELS)
plt.barh(y_pos, accuracies,  alpha=1)
plt.yticks(y_pos, labels)
plt.xlabel('Accuracy')
plt.title('List of categories and accuracy')
plt.show()

def eval_global(net, paths, labels, layer='probs'):
    top_labels = np.zeros((len(paths)))
    for i, path in enumerate(paths):
        processed_image = im = caffe.io.load_image(path)
        net.blobs['data'].data[...] = transformer.preprocess('data', processed_image)
        out = net.forward()
        top_labels[i] = net.blobs[layer].data[0].flatten().argsort()[-1]
        
    return np.count_nonzero(top_labels == labels) / float(len(paths)) * 100., top_labels



def show_confusion_matrix(true_labels, top_labels, labels):
    # confusion matrix
    matrix = confusion_matrix(true_labels, top_labels)
    # normalize confutsion matrix
    num_per_class , _ = np.histogram(test_labels, len(labels))
    for i, row in enumerate(matrix):
        matrix[i] = matrix[i] * 100.  / num_per_class[i]
    
    

    norm_conf = []
    for i in matrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure(figsize=(10,7))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.GnBu, 
                    interpolation='nearest')

    width, height = matrix.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(matrix[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = labels
    ax.xaxis.tick_top()
    plt.xticks(range(width), alphabet[:width], rotation='vertical')
    plt.yticks(range(height), alphabet[:height])

glob_accuracy, top_labels = eval_global(activitiesnet, test_paths, test_labels)

print "Global accuracy of the cnn:  %2.f%% " %(glob_accuracy)
show_confusion_matrix(test_labels, top_labels, labels)



