import tensorflow as tf
import numpy as np
from IPython.display import Image, display
from PIL import Image
import random
import os
import time
import math
import model
import preparation as prep
import predict as pred

dataset_dir = "../dataset/cropped"
logging_dir = '../log'
STEPS = 1000
LABELS = ['Hsia Yu-chiao', 'Sung Yun-hua']
TRAINING = True
PREDICTING = True

def training_placeholder(model_name, labelNames, t_batch_train, 
                         t_batch_evalu, log_train_dir, 
                         max_step, learning_rate):
    # Getting batch size and image size for placeholder
    batch_size, image_size, _, _ = t_batch_train[0].shape
    
    # Prepare place holder for training or evaluating dataset
    t_image_input = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3])
    t_label_input = tf.placeholder(tf.int32, shape=[batch_size])
    
    # Getting all training required operations
    t_op_logits, t_op_pred, classes = model.classification_inference(t_image_input, labelNames, model_name)
    t_op_loss = model.losses(t_op_logits, t_label_input)
    t_op_acc = model.evaluation(t_op_logits, t_label_input)
    t_op_train = model.training(t_op_loss, learning_rate)
    model.writeSummaries(t_op_acc, t_op_loss, scope='training')
    t_op_summary = tf.summary.merge_all()
    
    with tf.Session() as sess:
        # Summary for tensorboard
        summary_writer = tf.summary.FileWriter(logdir=log_train_dir, graph=sess.graph, filename_suffix='training')
        # Saver for saving model
        saver = tf.train.Saver()
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # Tensorflow Thread control
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        try:
            for step in range(1, max_step + 1):
                if coord.should_stop():
                    break
                
                tra_batch, val_batch = sess.run([t_batch_train, t_batch_evalu])                    
                _, tra_loss, tra_acc, summary_str =                     sess.run([t_op_train, t_op_loss, t_op_acc, t_op_summary],
                             feed_dict={t_image_input:tra_batch[0],
                                        t_label_input:tra_batch[1]})
                    
                summary_writer.add_summary(summary_str, step)
                if step % 100 == 0 or step == 1:
                    print('\n')
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                    print('', end='', flush=True)

                if step % 200 == 0 or step == max_step:
                    checkpoint_path = os.path.join(log_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                print('.', end='', flush=True)
                    
                
        except tf.errors.OutOfRangeError:
            print('Done training -- step limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

def training_preload(model_name, labelNames, t_batch_train, 
                     t_batch_evalu, log_train_dir, 
                     max_step, learning_rate):
    # Getting all training required operations
    t_op_logits, t_op_pred, classes = model.classification_inference(t_batch_train[0], labelNames, model_name)
    t_op_loss = model.losses(t_op_logits, t_batch_train[1])
    t_op_acc = model.evaluation(t_op_logits, t_batch_train[1])
    t_op_train = model.training(t_op_loss, learning_rate)

    model.writeSummaries(t_op_acc, t_op_loss, scope='training')    
    t_op_summary = tf.summary.merge_all()
        
    with tf.Session() as sess:
        # Summary for tensorboard
        summary_writer = tf.summary.FileWriter(logdir=log_train_dir, graph=sess.graph, filename_suffix='training')
        # Saver for saving model
        saver = tf.train.Saver()
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # Tensorflow Thread control
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        try:
            for step in range(1, max_step + 1):
                if coord.should_stop():
                    break

                _, tra_loss, tra_acc, summary_str = sess.run([t_op_train, t_op_loss, t_op_acc, t_op_summary])
                summary_writer.add_summary(summary_str, step)
                
                if step % 100 == 0 or step == 1:
                    print('\n')
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                    print('', end='', flush=True)
                    
                if step % 200 == 0 or step == max_step:
                    checkpoint_path = os.path.join(log_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
                print('.', end='', flush=True)
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- step limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

def run_training(train_dir, log_dir, model_name='Simple', img_format = 'jpg', image_size = 200,
                train_test_ratio = 0.2, batch_size = 64, capacity=1200,
                learning_rate = 0.001, max_step = 1000):
    log_model_dir = os.path.join(log_dir, model_name)
    log_train_dir = os.path.join(log_model_dir, "train")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_model_dir):
        os.makedirs(log_model_dir)

    tra_images,tra_labels,val_images,val_labels,labelNames = prep.readFilesInLabeledDirectory(train_dir, 0.2, img_format)
    number_classes = len(labelNames)
    print('Num of training data: {}\nNum of evaluateion data: {}\nNum of classes: {}'.format(len(tra_images), len(val_images), number_classes))

    train_batch = prep.get_distorted_files(tra_images,
                                         tra_labels,
                                         image_size,
                                         image_size,
                                         batch_size,
                                         capacity,
                                         True)
    
    val_batch = prep.get_distorted_files(val_images,
                                         val_labels,
                                         image_size,
                                         image_size,
                                         batch_size,
                                         capacity,
                                         False)
    
    # Training with preloading data
    training_preload(model_name, labelNames, train_batch, val_batch, log_train_dir,
                     max_step, learning_rate)
    
    # Training with placeholder and feed data later
#     training_placeholder(model_name, labelNames, train_batch, val_batch, log_train_dir,
#                          max_step, learning_rate)

MODEL_NAMES = ['Inception_v3', 'Resnet_v1', 'Resnet_v2', 'Cifar10', 'Simple', 'Custom']
def train_model(model_name):
    start = time.time()
    run_training(dataset_dir, logging_dir, model_name=model_name, img_format='png',
                 max_step=STEPS, learning_rate=0.001)
    end = time.time()
    return end - start

model_name = 'Simple'
if (TRAINING):
    costs = train_model(model_name)

if (PREDICTING):
    prediction_dir = "../dataset/predicting/cropped"
#     prediction_dir = "./training_pics/batches_in_step_100"
    original_image = "../dataset/predicting/unknown/123.jpg"
    ori = Image.open(original_image)
    display(ori)
    tmp_imgs = []
    for file in os.listdir(prediction_dir):
        if file.endswith('png'):
            img = Image.open(os.path.join(prediction_dir, file))
            arr = np.array(img)
            tmp_imgs.append(arr)
    model_dir = '../log/' + model_name + '/train'
    results, labels = pred.classify_images(model_name, model_dir, tmp_imgs, labels=LABELS)

for idx, file in enumerate(os.listdir(prediction_dir)):
    if file.endswith('png'):
        img = Image.open(os.path.join(prediction_dir, file))
        display(img)
        print(file)
        print('Hsia Yu-chiao: {}, Sung Yun-hua:{}'.format(results[idx - 1][0], results[idx - 1][1]) )









