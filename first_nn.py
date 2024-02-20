# Phillip Waul
# 2/20/24
# This is a barebones CNN where we just have it go over our dataset with little to no real changes.
# first neural network with keras tutorial
import numpy as np
# mlp for the blobs multi-class classification problem with cross-entropy loss
import pandas as pd
import keras
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf



# Example: NumPy arrays
(train_images, train_labels), (test_images, test_labels) = keras.utils.image_dataset_from_directory(
    directory="images",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=.5,
    subset="both",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    data_format=None,
)


print(train_labels)

y = to_categorical(reader_mix['Most_Frequent_Answer'])
print(reader_mix)
print(y)


#TODO change this so that we use our data from reproductive_rtu instead of mnist
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#train_images = train_images.reshape((60000, 28, 28, 1))
#train_images_val = train_images[:10000]
#partial_train_images = train_images[10000:]
#train_labels = to_categorical(train_labels)
#test_labels = to_categorical(test_labels)
#######train_labels_val = train_labels[:10000]
#partial_train_labels=train_labels[10000:]
#train_images_val = train_images_val.astype('float32')/255
#partial_train_images=partial_train_images.astype('float32')/255
#test_images = test_images.reshape((10000, 28, 28, 1))
#test_images = test_images.astype('float32') / 255