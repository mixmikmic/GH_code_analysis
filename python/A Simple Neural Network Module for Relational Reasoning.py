import time
from tqdm import *
from sort_of_clevr_generator import SortOfCLEVRGenerator
import cPickle as pickle

generator = SortOfCLEVRGenerator()
test_dataset = []
train_dataset = []
test_size = 200 #200
train_size = 9800 #9800

try:
    filename = 'sort-of-clevr.p'
    with open(filename, 'rb') as f:
        train_dataset, test_dataset = pickle.load(f)

except:

    for i in tqdm(xrange(train_size), desc='Generating Sort-of-CLEVR Training Dataset'):
        dataset = generator.generate_dataset()
        train_dataset.append(dataset)

    for i in tqdm(xrange(test_size), desc='Generating Sort-of-CLEVR Test Dataset'):
        dataset = generator.generate_dataset()
        test_dataset.append(dataset)
    
    with open("sort-of-clevr.p", 'wb') as f:
        pickle.dump((train_dataset, test_dataset), f, protocol=2)

import numpy as np
train_img = []
train_q = []
train_a = []
for img, questions, answers in test_dataset:
    img_train = img/255
    for q, a in zip(questions, answers):
        train_img += [img_train]
        train_q += [q]
        train_a += [a]
train_img = np.stack(train_img)
train_q = np.vstack(train_q)
train_a = np.vstack(train_a)

from sort_of_clevr_generator import SortOfCLEVRGenerator
generator = SortOfCLEVRGenerator()
img, questions, answers = generator.generate_dataset()
print img.nbytes
img.dtype

import matplotlib.pyplot as plt

def visualize_img(img):
    img = np.fliplr(img.reshape(-1,3)).reshape(img.shape)
    plt.imshow(img)
    plt.show()

def translate_question(q):
    if len(q) != 11:
        return 'Not a proper question'
    colors = ['red', 'blue', 'green', 'orange', 'yellow', 'gray']
    idx= np.argwhere(q[:6])[0][0]
    color = colors[idx]
    if q[6]:
        if q[8]:
            return 'The shape of the nearest object to the object in ' + color + ' is?' 
        elif q[9]:
            return 'The shape of the farthest object away from the object in ' + color + ' is?'
        elif q[10]:
            return 'How many objects have the same shape as the object in ' + color + '?'
    else:
        if q[8]:
            return 'Is the object in color ' + color + ' a circle or a rectangle?'
        elif q[9]:
            return 'Is the object in color ' + color + ' on the bottom of the image?'
        elif q[10]:
            return 'Is the object in color ' + color + ' on the left of the image?'
        
def translate_answer(a):
    if len(a) != 10:
        return 'Not a proper answer'
    if a[0]:
        return 'yes'
    if a[1]:
        return 'no'
    if a[2]:
        return 'rectangle'
    if a[3]:
        return 'circle'
    return np.argwhere(a[4:])[0][0] + 1

import numpy as np
idx = 20
idb = 10
img, q, a = train_dataset[idx]
visualize_img(img/255)
print img.shape
print translate_question(q[idb])
print translate_answer(a[idb])

import keras
from keras.layers.convolutional import Conv2D

def ConvolutionNetworks(kernel_size=3, stride_size=2):
    def conv(model):
        model = Conv2D(24, (5, 5), strides=(stride_size, stride_size),activation='relu',input_shape=(75, 75, 3), data_format='channels_last')(model)
        model = BatchNormalization()(model)
        model = Conv2D(24, (5, 5), strides=(stride_size, stride_size),activation='relu')(model)
        model = BatchNormalization()(model)
        model = Conv2D(24, (kernel_size, kernel_size), strides=(stride_size, stride_size),activation='relu')(model)
        model = BatchNormalization()(model)
        model = Conv2D(24, (3, 3), strides=(1, 1),activation='relu')(model)
        model = BatchNormalization()(model)
        return model
    return conv

import numpy as np
import keras
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Concatenate, Add
from keras.layers.normalization import BatchNormalization


def slicer(x_loc, y_loc):
    def func(x):
        return x[:,x_loc,y_loc,:]
    return Lambda(func)

def object_tagging(o, i, d):
    coor = K.variable(value=[float(int(i/d))/d*2-1, float((i%d))/d*2-1])
    coor = K.expand_dims(coor, axis=0)
    batch_size = K.shape(o)[0]
    coor = K.tile(coor, [batch_size, 1])
    coor = Input(tensor=coor)
    o = Concatenate()([coor, o])
    return o
    
def compute_relations(objects, question):
    
    ##############################################
    # The code here is inspired by Alan Lee in https://github.com/Alan-Lee123/relation-network/blob/master/train.py
    # My original code needes too much memory to run because I required one lambda layer for each of the d^2 objects
    # however, in Alan Lee's code he was able to accombplish that with only 4.
    
    def get_top_dim_1(t):
        return t[:, 0, :, :]

    def get_all_but_top_dim_1(t):
        return t[:, 1:, :, :]

    def get_top_dim_2(t):
        return t[:, 0, :]

    def get_all_but_top_dim2(t):
        return t[:, 1:, :]
    
    slice_top_dim_1 = Lambda(get_top_dim_1)
    slice_all_but_top_dim_1 = Lambda(get_all_but_top_dim_1)
    slice_top_dim_2 = Lambda(get_top_dim_2)
    slice_all_but_top_dim2 = Lambda(get_all_but_top_dim2)
    
    d = K.int_shape(objects)[2]
    features = []
    for i in range(d):
        features1 = slice_top_dim_1(objects)
        objects = slice_all_but_top_dim_1(objects)
        for j in range(d):
            features2 = slice_top_dim_2(features1)
            features1 = slice_all_but_top_dim2(features1)
            features.append(features2)
    
    relations = []
    concat = Concatenate()
    for feature1 in features:
        for feature2 in features:
            relations.append(concat([feature1, feature2, question]))
    
    ##############################################
#     relations = []
#     #objects are tagged CNN output that has the format of (batch_size, 24+2(tagging), d, d)
#     d = K.int_shape(objects)[2]
#     for i in xrange(d*d):
#         o_i = slicer(int(i / d), int(i % d))(objects)
# #         o_i = object_tagging(o_i, i, d)
#         for j in xrange(d*d):
#             o_j = slicer(int(j / d), int(j % d))(objects)
# #             o_j = object_tagging(o_j, j, d)
#             relations.append(Concatenate()([o_i, o_j, question]))
    return relations

from keras.models import Model
from keras.layers import Input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

def f_theta():
    def f(model):
        model = Dense(256)(model)
        model = Activation('relu')(model)
        model = Dense(256)(model)
        model = Activation('relu')(model)
#         model = Dropout(0.5)(model)
        model = Dense(256)(model)
        model = Activation('relu')(model)
        model = Dense(256)(model)
        model = Activation('relu')(model)
        return model
    return f

baseline_scene = Input((75, 75, 3))
baseline_question = Input((11,))
baseline_conv = ConvolutionNetworks()(baseline_scene)
baseline_conv = Flatten()(baseline_conv)
baseline_conv = Concatenate()([baseline_conv, baseline_question])
baseline_output = f_theta()(baseline_conv) 
baseline_output = Dense(10, activation='softmax')(baseline_output)
BaseLineModel = Model(inputs=[baseline_scene, baseline_question], outputs=baseline_output)
SVG(model_to_dot(BaseLineModel, show_shapes=True).create(prog='dot', format='svg'))

from keras.utils import plot_model

def g_th(layers):
    def f(model):
        for n in xrange(len(layers)):
            model = layers[n](model)
        return model
    return f

def stack_layer(layers):
    def f(x):
        for k in range(len(layers)):
            x = layers[k](x)
        return x
    return f

def g_theta(h_unit=256, layers=4):
    r = []
    for k in xrange(layers):
        r.append(Dense(h_unit))
        r.append(Activation('relu'))
    return g_th(r)

def get_MLP():
    return g_th()

def RelationNetworks(objects, question):
    g_t = g_theta()
    relations = compute_relations(objects,question)
    print len(relations)
    g_all = []
    for i, r in enumerate(relations):
        g_all.append(g_t(r))
    combined_relation = Add()(g_all)
    f_out = f_theta()(combined_relation)
    return f_out

def build_tag(conv):
    d = K.int_shape(conv)[2]
    tag = np.zeros((d,d,2))
    for i in xrange(d):
        for j in xrange(d):
            tag[i,j,0] = float(int(i%d))/(d-1)*2-1
            tag[i,j,1] = float(int(j%d))/(d-1)*2-1
    tag = K.variable(tag)
    tag = K.expand_dims(tag, axis=0)
    batch_size = K.shape(conv)[0]
    tag = K.tile(tag, [batch_size,1,1,1])
    return Input(tensor=tag)

visual_scene = Input((75, 75, 3))
visual_question = Input((11,))
visual_conv = ConvolutionNetworks()(visual_scene)
tag = build_tag(visual_conv)
visual_conv = Concatenate()([tag, visual_conv])
visual_RN = RelationNetworks(visual_conv, visual_question)
visual_out = Dense(10, activation='softmax')(visual_RN)
VisualModel = Model(inputs=[visual_scene, visual_question, tag], outputs=visual_out)
plot_model(VisualModel, to_file='figures/VisualModel1.png')

plot_model(VisualModel, to_file='figures/VisualModel1.png')

SVG(model_to_dot(VisualModel).create(prog='dot', format='svg'))

import numpy as np

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, training_set, test_set, is_baseline, dim_x = 75, dim_y = 75, channel = 3, q_dim = 11, a_dim = 10, batch_size = 64, shuffle = True):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.channel = channel
        self.q_dim = q_dim
        self.a_dim = a_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training_set = training_set
        self.test_set = test_set
        self.is_baseline = is_baseline

    def generate_training(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            
            if self.shuffle:
                np.random.shuffle(self.training_set)

            # Generate batches
            data_size = len(self.training_set)
            imax = int(data_size/self.batch_size)
            for i in range(imax):
                # Generate data
                imgs, questions, answers = self.__data_generation(self.training_set[i: i + self.batch_size])
                imgs, questions, answers = self.randomize(imgs, questions, answers)
                yield [imgs, questions], answers
                
    def generate_test(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            if self.shuffle:
                np.random.shuffle(self.test_set)

            # Generate batches
            data_size = len(self.test_set)
            imax = int(data_size/self.batch_size)
            for i in range(imax):
                # Generate data
                imgs, questions, answers = self.__data_generation(self.test_set[i: i+self.batch_size])
                imgs, questions, answers = self.randomize(imgs, questions, answers)
                yield [imgs, questions], answers
    
    def randomize(self, a, b, c):
        # Generate the permutation index array.
        permutation = np.random.permutation(a.shape[0])
        # Shuffle the arrays by giving the permutation in the square brackets.
        shuffled_a = a[permutation]
        shuffled_b = b[permutation]
        shuffled_c = c[permutation]
        return shuffled_a, shuffled_b, shuffled_c

    def __data_generation(self, dataset):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        q_lenght = len(dataset[0][1])
        imgs = np.empty((self.batch_size*q_lenght, self.dim_x, self.dim_y, self.channel))
        questions = np.empty((self.batch_size*q_lenght, self.q_dim), dtype = int)
        answers = np.empty((self.batch_size*q_lenght, self.a_dim), dtype = int)
        c = 0
        for img, question, answer in dataset:
            img = img/255
            for q, a in zip(question, answer):
                imgs[c, :, :, :]  = img
                questions[c, :] = q
                answers[c, :] = a
                c += 1
        return imgs, questions, answers

from keras.optimizers import Adam
lr = 1e-4
adam = Adam(lr=lr)
VisualModel.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

batch_size = 64
data_generator = DataGenerator(train_dataset, test_dataset, False, batch_size=batch_size)
training_generator = data_generator.generate_training()
validation_generator = data_generator.generate_test()
visualmodel_history = VisualModel.fit_generator(generator = training_generator,
                    steps_per_epoch = (len(train_dataset))//batch_size,
                    validation_data = validation_generator,
                    validation_steps = (len(test_dataset))//batch_size,
                    epochs = 5)

VisualModel.save('models/VisualModelLarge5.h5')

from keras import models
VisualModel.load_weights('models/VisualModelLarge.h5')

import matplotlib.pyplot as plt

plt.plot(visualmodel_history.history['acc'])
plt.plot(visualmodel_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('figures/VisualModelAccuracy')
plt.show()

from keras.optimizers import Adam
lr = 1e-4
adam = Adam(lr=lr)
BaseLineModel.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

data_generator = DataGenerator(train_dataset, test_dataset, False)
training_generator = data_generator.generate_training()
validation_generator = data_generator.generate_test()
baseline_history = BaseLineModel.fit_generator(generator = training_generator,
                    steps_per_epoch = (len(train_dataset)*20)//64,
                    validation_data = validation_generator,
                    validation_steps = (len(test_dataset)*20)//64,
                    epochs = 50)
BaseLineModel.save('BaseLineModel.h5')

BaseLineModel.save('BaseLineModel.h5')

import matplotlib.pyplot as plt

plt.plot(baseline_history.history['acc'])
plt.plot(baseline_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('figures/BaselineAccuracy')
plt.show()

from sort_of_clevr_generator import SortOfCLEVRGenerator

def create_test_set(dataset_size):
    generator = SortOfCLEVRGenerator()
    testset = []
    rel_testset = []
    norel_testset = []
    norel_ids = []
    rel_ids = []

    for i in xrange(dataset_size):
        imgs, questions, answers = generator.generate_dataset()
        testset.append((imgs, questions, answers))
        norel_questions = []
        rel_questions = []
        norel_answers = []
        rel_answers = []
        for q_idx in xrange(len(questions)):
            if questions[q_idx][6] == 1:
                rel_questions.append(questions[q_idx])
                rel_answers.append(answers[q_idx]) 
            else:
                norel_questions.append(questions[q_idx])
                norel_answers.append(answers[q_idx]) 
        norel_testset.append((imgs, norel_questions, norel_answers))
        rel_testset.append((imgs, rel_questions, rel_answers))
    return rel_testset, norel_testset, testset

def test_model(model):
    rel, norel, test = create_test_set(200)
    batch_size = 64
    rel_generator = DataGenerator(None, rel, False, batch_size=batch_size)
    norel_generator = DataGenerator(None, norel, False, batch_size=batch_size)
    test_generator = DataGenerator(None, test, False, batch_size=batch_size)
    r_generator = rel_generator.generate_test()
    res_rel = model.evaluate_generator(r_generator, steps=len(rel)//batch_size, max_queue_size=10, workers=1, use_multiprocessing=False)
    nr_generator = norel_generator.generate_test()
    res_norel = model.evaluate_generator(nr_generator, steps=len(rel)//batch_size, max_queue_size=10, workers=1, use_multiprocessing=False)
    re_generator = test_generator.generate_test()
    res = model.evaluate_generator(re_generator, steps=len(rel)//batch_size, max_queue_size=10, workers=1, use_multiprocessing=False)
    print res_rel
    print res_norel
    print res
    return res_rel[1], res_norel[1], res[1]

res = test_model(VisualModel)

print ('The accracy of Relational Question is %f, The accuracy of Non-Relational Question is %f, Overall Accuracy is %f' %res)

from keras import models
BaseLineModel = models.load_model('models/BaseLineModel.h5')

res = test_model(BaseLineModel)

print ('The accracy of Relational Question is %f, The accuracy of Non-Relational Question is %f, Overall Accuracy is %f' %res)

