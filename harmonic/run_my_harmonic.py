"""Run MNIST-rot"""

import argparse
import math
import os
import os
import random
import sys
sys.path.append("../")
import time
from PIL import Image
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from my_harmonic_model import my_harmonic

# This is unnecessary for our project, but it was in the original so I will leave it here for now.
# def download2FileAndExtract(url, folder, fileName):
#    print('Downloading rotated MNIST...')
#    add_folder(folder)
#    zipFileName = folder + fileName
#    request = urllib2.urlopen(url)
#    with open(zipFileName, "wb") as f :
#       f.write(request.read())
#    if not zipfile.is_zipfile(zipFileName):
#       print('ERROR: ' + zipFileName + ' is not a valid zip file.')
#       sys.exit(1)
#    print('Extracting...')
#    wd = os.getcwd()
#    os.chdir(folder)

#    archive = zipfile.ZipFile('.'+fileName, mode='r')
#    archive.extractall()
#    archive.close()
#    os.chdir(wd)
#    print('Successfully retrieved rotated rotated MNIST dataset.')

def load_images_from_directory(directory, image_size):
   images = []
   labels = []
   for label_dir in os.listdir(directory):
      if label_dir == ".DS_Store":
         continue
      i = 0
      label = i #int(label_dir)
      label_path = os.path.join(directory, label_dir)
      for image_file in os.listdir(label_path):
         image_path = os.path.join(label_path, image_file)
         # Even learnable resizing requires an initial resizing.
         image = Image.open(image_path).resize(image_size, Image.BILINEAR )
         image = image.convert('RGB')
         image = np.array(image, dtype="float32")  # Normalize the image
         images.append(image)
         labels.append(label)
      i += 1
   return np.array(images, dtype="float32"), np.array(labels)


def getData(args):
   img_w, img_h = 256, 256
   print(os.getcwd())
   images, labels = load_images_from_directory("./images", (img_w, img_h))
   
   #dividing images into a testing set and a trainging set
   testing_ratio = .3
   floor = math.floor(len(images)*testing_ratio)
   divider = math.floor(len(images)*testing_ratio)
   train_images = images[:-divider]
   test_images = images[-divider-1:]
   divider = math.floor(len(labels)*testing_ratio)
   train_labels = labels[:-divider]
   test_labels = labels[-divider-1:]

   
   data = {}
   if args.combine_train_val:
      data['train_x'] = train_images
      data['train_y'] = train_labels
   else:
      val_ratio = .2
      divider = math.floor(len(train_images)*val_ratio)
      data['train_x'] = train_images[:-divider]
      data['train_y'] = train_labels[:-divider]
      data['valid_x'] = train_images[-divider-1:]
      data['valid_y'] = train_labels[-divider-1:]
   data['test_x'] = test_images
   data['test_y'] = test_labels
   return data

def settings(args):
   # Download MNIST if it doesn't exist
   # args.dataset = 'rotated_mnist'
   # if not os.path.exists(args.data_dir + '/mnist_rotation_new.zip'):
   #    download2FileAndExtract("https://www.dropbox.com/s/0fxwai3h84dczh0/mnist_rotation_new.zip?dl=1",
   #       args.data_dir, "/mnist_rotation_new.zip")
   # # Load dataset
   # mnist_dir = args.data_dir + '/mnist_rotation_new'
   # train = np.load(mnist_dir + '/rotated_train.npz')
   # valid = np.load(mnist_dir + '/rotated_valid.npz')
   # test = np.load(mnist_dir + '/rotated_test.npz')
   # data = {}
   # if args.combine_train_val:
   #    data['train_x'] = np.vstack((train['x'], valid['x']))
   #    data['train_y'] = np.hstack((train['y'], valid['y']))
   # else:
   #    data['train_x'] = train['x']
   #    data['train_y'] = train['y']
   #    data['valid_x'] = valid['x']
   #    data['valid_y'] = valid['y']
   # data['test_x'] = test['x']
   # data['test_y'] = test['y']
   data = getData(args) 
   
   # Other options
   if args.default_settings:
      args.n_epochs = 200
      args.batch_size = 46
      args.learning_rate = 0.0076
      args.std_mult = 0.7
      args.delay = 12
      args.phase_preconditioner = 7.8
      args.filter_gain = 2
      args.filter_size = 5
      args.n_rings = 4
      args.n_filters = 8
      args.display_step = len(data['train_x'])/46
      args.is_classification = True
      args.dim = 28
      args.crop_shape = 0
      args.n_channels = 1
      args.n_classes = 10
      args.lr_div = 10.

   args.log_path = add_folder('./logs')
   args.checkpoint_path = add_folder('./checkpoints') + '/model.ckpt'
   return args, data

def add_folder(folder_name):
   if not os.path.exists(folder_name):
      os.mkdir(folder_name)
      print('Created {:s}'.format(folder_name))
   return folder_name

def minibatcher(inputs, targets, batchsize, shuffle=False):
   assert len(inputs) == len(targets)
   if shuffle:
      indices = np.arange(len(inputs))
      np.random.shuffle(indices)
   for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
      if shuffle:
         excerpt = indices[start_idx:start_idx + batchsize]
      else:
         excerpt = slice(start_idx, start_idx + batchsize)
      yield inputs[excerpt], targets[excerpt]

def get_learning_rate(args, current, best, counter, learning_rate):
   """If have not seen accuracy improvement in delay epochs, then divide 
   learning rate by 10
   """
   if current > best:
      best = current
      counter = 0
   elif counter > args.delay:
      learning_rate = learning_rate / args.lr_div
      counter = 0
   else:
      counter += 1
   return (best, counter, learning_rate)

def main(args):
   """The magic happens here"""
   tf.compat.v1.reset_default_graph()
   ##### SETUP AND LOAD DATA #####
   args, data = settings(args)
   ##### BUILD MODEL #####
   ## Placeholders
   # [args.batch_size, 784]
   x = tf.compat.v1.placeholder(tf.float32, None, name='x')
   y = tf.compat.v1.placeholder(tf.int64, [args.batch_size], name='y')
   learning_rate = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
   train_phase = tf.compat.v1.placeholder(tf.bool, name='train_phase')

   # Construct model and optimizer
   pred = my_harmonic(args, x, train_phase)
   loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))

   # Evaluation criteria
   correct_pred = tf.equal(tf.argmax(pred, 1), y)
   accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

   # Optimizer
   optim = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
   grads_and_vars = optim.compute_gradients(loss)
   modified_gvs = []
   # We precondition the phases, for faster descent, in the same way as biases
   for g, v in grads_and_vars:
      if 'psi' in v.name:
         g = args.phase_preconditioner*g
      modified_gvs.append((g, v))
   train_op = optim.apply_gradients(modified_gvs)
   
   ##### TRAIN ####
   # Configure tensorflow session
   init_global = tf.compat.v1.global_variables_initializer()
   init_local = tf.compat.v1.local_variables_initializer()
   config = tf.compat.v1.ConfigProto()
   config.gpu_options.allow_growth = True
   config.log_device_placement = False
   
   lr = args.learning_rate
   saver = tf.compat.v1.train.Saver()
   sess = tf.compat.v1.Session(config=config)
   sess.run([init_global, init_local], feed_dict={train_phase : True})
   
   start = time.time()
   epoch = 0
   step = 0.
   counter = 0
   best = 0.
   print('Starting training loop...')
   while epoch < args.n_epochs:
      # Training steps
      batcher = minibatcher(data['train_x'], data['train_y'], args.batch_size, shuffle=True)
      train_loss = 0.
      train_acc = 0.
      for i, (X, Y) in enumerate(batcher):
         feed_dict = {x: X, y: Y, learning_rate: lr, train_phase: True}
         __, loss_, accuracy_ = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)
         train_loss += loss_
         train_acc += accuracy_
         sys.stdout.write('{:d}/{:d}\r'.format(i, data['train_x'].shape[0]/args.batch_size))
         sys.stdout.flush()
      train_loss /= (i+1.)
      train_acc /= (i+1.)
      
      if not args.combine_train_val:
         batcher = minibatcher(data['valid_x'], data['valid_y'], args.batch_size)
         valid_acc = 0.
         for i, (X, Y) in enumerate(batcher):
            feed_dict = {x: X, y: Y, train_phase: False}
            accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
            valid_acc += accuracy_
            sys.stdout.write('Validating\r')
            sys.stdout.flush()
         valid_acc /= (i+1.)
         print('[{:04d} | {:0.1f}] Loss: {:04f}, Train Acc.: {:04f}, Validation Acc.: {:04f}, Learning rate: {:.2e}'.format(epoch,
            time.time()-start, train_loss, train_acc, valid_acc, lr))
      else:
         print('[{:04d} | {:0.1f}] Loss: {:04f}, Train Acc.: {:04f}, Learning rate: {:.2e}'.format(epoch,
            time.time()-start, train_loss, train_acc, lr))
       # Save model     
      if epoch % 10 == 0:
         saver.save(sess, args.checkpoint_path)
         print('Model saved')
      
      # Updates to the training scheme
      #best, counter, lr = get_learning_rate(args, valid_acc, best, counter, lr)
      lr = args.learning_rate * np.power(0.1, epoch / 50)
      epoch += 1

   # TEST
   batcher = minibatcher(data['test_x'], data['test_y'], args.batch_size)
   test_acc = 0.
   for i, (X, Y) in enumerate(batcher):
      feed_dict = {x: X, y: Y, train_phase: False}
      accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
      test_acc += accuracy_
      sys.stdout.write('Testing\r')
      sys.stdout.flush()
   test_acc /= (i+1.)
   
   print('Test Acc.: {:04f}'.format(test_acc))
   sess.close()

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--data_dir", help="data directory", default='./data')
   parser.add_argument("--default_settings", help="use default settings", type=bool, default=True)
   parser.add_argument("--combine_train_val", help="combine the training and validation sets for testing", type=bool, default=False)
   main(parser.parse_args())