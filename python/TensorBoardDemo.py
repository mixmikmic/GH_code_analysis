from keras.models import Model
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from keras.layers import Input, Dropout
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import SGD

import tensorflow as tf

from settings import *

import numpy as np
import os
import dataset
from dataset import MyDataset

db=MyDataset(feature_dir=os.path.join('./IRMAS-Sample', 'features', 'Training'), batch_size=8, time_context=128, step=50, 
             suffix_in='_mel_',suffix_out='_label_',floatX=np.float32,train_percent=0.8)
val_data = db()

def build_model(n_classes):

    input_shape = (N_MEL_BANDS, SEGMENT_DUR, 1)
    channel_axis = 3
    melgram_input = Input(shape=input_shape)

    m_size = 70
    n_size = 3
    n_filters = 64
    maxpool_const = 4

    x = Convolution2D(n_filters, (m_size, n_size),
                      padding='same',
                      kernel_initializer='zeros',
                      kernel_regularizer=l2(1e-5))(melgram_input)

    x = BatchNormalization(axis=channel_axis)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(N_MEL_BANDS, SEGMENT_DUR/maxpool_const))(x)
    x = Flatten()(x)

    x = Dropout(0.5)(x)
    x = Dense(n_classes, kernel_initializer='zeros', kernel_regularizer=l2(1e-5), 
              activation='softmax', name='prediction')(x)

    model = Model(melgram_input, x)

    return model

model = build_model(IRMAS_N_CLASSES)

init_lr = 0.001
optimizer = SGD(lr=init_lr, momentum=0.9, nesterov=True)

model.summary()
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(db,
                    steps_per_epoch=4,
                    epochs=4,
                    verbose=2,
                    validation_data=val_data,
                    class_weight=None,
                    workers=1)

from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard

early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_EPOCH)
save_clb = ModelCheckpoint("{weights_basepath}/".format(weights_basepath=MODEL_WEIGHT_BASEPATH) +
                           "epoch.{epoch:02d}-val_loss.{val_loss:.3f}",
                           monitor='val_loss',
                           save_best_only=True)

tb = TensorBoard(log_dir='./example_1',
                 write_graph=True, write_grads=True, 
                 write_images=True, histogram_freq=1)
# if we want to compute activations and weight histogram, we need to specify the validation data for that. 
tb.validation_data = val_data

model.fit_generator(db,
                    steps_per_epoch=1, # change to STEPS_PER_EPOCH
                    epochs=1, # change to MAX_EPOCH_NUM
                    verbose=2,
                    validation_data=val_data,
                    callbacks=[save_clb, early_stopping, tb],
                    class_weight=None,
                    workers=1)

def build_model(n_classes):

    input_shape = (N_MEL_BANDS, SEGMENT_DUR, 1)
    channel_axis = 3
    melgram_input = Input(shape=input_shape)

    m_size = 70
    n_size = 3
    n_filters = 64
    maxpool_const = 4

    x = Convolution2D(n_filters, (m_size, n_size),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-5))(melgram_input)

    x = BatchNormalization(axis=channel_axis)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(N_MEL_BANDS, SEGMENT_DUR/maxpool_const))(x)
    x = Flatten()(x)

    x = Dropout(0.5)(x)
    x = Dense(n_classes, kernel_initializer='he_normal', kernel_regularizer=l2(1e-5), 
              activation='softmax', name='prediction')(x)

    model = Model(melgram_input, x)

    return model

model = build_model(IRMAS_N_CLASSES)

global_namescope = 'train'

def build_model(n_classes):

    with tf.name_scope('input'):
        input_shape = (N_MEL_BANDS, SEGMENT_DUR, 1)
        channel_axis = 3
        melgram_input = Input(shape=input_shape)

        m_size = [5, 5]
        n_size = [5, 5]
        n_filters = 64
        maxpool_const = 8

    with tf.name_scope('conv1'):
        x = Convolution2D(n_filters, (m_size[0], n_size[0]),
                          padding='same',
                          kernel_initializer='he_uniform')(melgram_input)
        x = BatchNormalization(axis=channel_axis)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(maxpool_const, maxpool_const))(x)

    with tf.name_scope('conv2'):
        x = Convolution2D(n_filters*2, (m_size[1], n_size[1]),
                          padding='same',
                          kernel_initializer='he_uniform')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(maxpool_const, maxpool_const))(x)
        x = Flatten()(x)

    with tf.name_scope('dense1'):
        x = Dropout(0.5)(x)
        x = Dense(n_filters, kernel_initializer='he_uniform', name='hidden')(x)
        x = ELU()(x)

    with tf.name_scope('dense2'):
        x = Dropout(0.5)(x)
        x = Dense(n_classes, kernel_initializer='he_uniform', activation='softmax', name='prediction')(x)

    model = Model(melgram_input, x)

    return model

model = build_model(IRMAS_N_CLASSES)

with tf.name_scope('optimizer'):
    optimizer = SGD(lr=init_lr, momentum=0.9, nesterov=True)

with tf.name_scope('model'):
    model = build_model(IRMAS_N_CLASSES)

# for the sake of memory, only graphs now
with tf.name_scope('callbacks'):
    # The TensorBoard developers are strongly encourage us to use different directories for every run
    tb = TensorBoard(log_dir='./example_3', write_graph=True)

# yes, we need to recompile the model every time
with tf.name_scope('compile'):
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# and preudo-train the model
with tf.name_scope(global_namescope):
    model.fit_generator(db,
                        steps_per_epoch=1, # just one step
                        epochs=1, # one epoch to save the graphs
                        verbose=2,
                        validation_data=val_data,
                        callbacks=[tb],
                        workers=1)

from keras import backend as K
if K.backend() == 'tensorflow':
    import tensorflow as tf
    from tensorflow.contrib.tensorboard.plugins import projector

class TensorBoardHiddenOutputVis(Callback):
    """Tensorboard Intermediate Outputs visualization callback."""

    def __init__(self, log_dir='./logs_embed',
                 batch_size=32,
                 freq=0,
                 layer_names=None,
                 metadata=None,
                 sprite=None,
                 sprite_shape=None):
        super(TensorBoardHiddenOutputVis, self).__init__()
        self.log_dir = log_dir
        self.freq = freq
        self.layer_names = layer_names
        # Notice, that only one file is supported in the present callback
        self.metadata = metadata
        self.sprite = sprite
        self.sprite_shape = sprite_shape
        self.batch_size = batch_size

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        self.summary_writer = tf.summary.FileWriter(self.log_dir)
        self.outputs_ckpt_path = os.path.join(self.log_dir, 'keras_outputs.ckpt')

        if self.freq and self.validation_data:
            # define tensors to compute outputs on
            outputs_layers = [layer for layer in self.model.layers
                                 if layer.name in self.layer_names]
            self.output_tensors = [tf.get_default_graph().get_tensor_by_name(layer.get_output_at(0).name)
                                   for layer in outputs_layers]

            # create configuration for visualisation in the same manner as for embeddings
            config = projector.ProjectorConfig()
            for i in range(len(self.output_tensors)):
                embedding = config.embeddings.add()
                embedding.tensor_name = '{ns}/hidden_{i}'.format(ns=global_namescope, i=i)

                # Simpliest metadata handler, a single file for all embeddings
                if self.metadata:
                    embedding.metadata_path = self.metadata

                # Sprite image handler
                if self.sprite and self.sprite_shape:
                    embedding.sprite.image_path = self.sprite
                    embedding.sprite.single_image_dim.extend(self.sprite_shape)

            # define TF variables to store the hidden outputs during the training
            # Notice, that only 1D outputs are supported
            self.hidden_vars = [tf.Variable(np.zeros((len(self.validation_data[0]),
                                                         self.output_tensors[i].shape[1]),
                                                        dtype='float32'),
                                               name='hidden_{}'.format(i))
                                   for i in range(len(self.output_tensors))]
            # add TF variables into computational graph
            for hidden_var in self.hidden_vars:
                self.sess.run(hidden_var.initializer)

            # save the config and setup TF saver for hidden variables
            projector.visualize_embeddings(self.summary_writer, config)
            self.saver = tf.train.Saver(self.hidden_vars)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data and self.freq:
            if epoch % self.freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)
                all_outputs = [[]]*len(self.output_tensors)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                # compute outputs batch by batch on validation data
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    batch_val = []
                    batch_val.append(val_data[0][i:i + step])
                    batch_val.append(val_data[1][i:i + step])
                    batch_val.append(val_data[2][i:i + step])
                    if self.model.uses_learning_phase:
                        batch_val.append(val_data[3])
                    feed_dict = dict(zip(tensors, batch_val))
                    tensor_outputs = self.sess.run(self.output_tensors, feed_dict=feed_dict)
                    for output_idx, tensor_output in enumerate(tensor_outputs):
                        all_outputs[output_idx].extend(tensor_output)
                    i += self.batch_size
                
                # rewrite the current state of hidden outputs with new values
                for idx, embed in enumerate(self.hidden_vars):
                    embed.assign(np.array(all_outputs[idx])).eval(session=self.sess)
                self.saver.save(self.sess, self.outputs_ckpt_path, epoch)

        self.summary_writer.flush()

    def on_train_end(self, _):
        self.summary_writer.close()

layers_to_monitor = ['hidden']
# find the files precomputed in ./logs_embed directory 
metadata_file_name = 'metadata.tsv'
sprite_file_name = 'sprite.png'
sprite_shape = [N_MEL_BANDS, SEGMENT_DUR]

with tf.name_scope('callbacks'):
    tbe = TensorBoardHiddenOutputVis(log_dir='./logs_embed', freq=1,
                           layer_names=layers_to_monitor,
                           metadata=metadata_file_name,
                           sprite=sprite_file_name,
                           sprite_shape=sprite_shape)
    tbe.validation_data = val_data

with tf.name_scope('compile'):
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

with tf.name_scope(global_namescope):
    model.fit_generator(db,
                        steps_per_epoch=1, # change to STEPS_PER_EPOCH
                        epochs=1, # change to MAX_EPOCH_NUM
                        verbose=2,
                        callbacks=[tbe],
                        validation_data=val_data,
                        class_weight=None,
                        workers=1)



