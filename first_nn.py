# Phillip Waul
# 2/20/24
# This is a barebones CNN where we just have it go over our dataset with little to no real changes.
# first neural network with keras tutorial

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

# Load the dataset from the directory
train_dataset = image_dataset_from_directory(
    directory="images",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,  # Seed for shuffling and transformations
    validation_split=0.2,  # Adjust the split ratio as needed
    subset="training",
    interpolation="bilinear",
)

validation_dataset = image_dataset_from_directory(
    directory="images",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,  # Use the same seed as for the training dataset
    validation_split=0.2,  # Adjust the split ratio as needed
    subset="validation",
    interpolation="bilinear",
)

# Example of how to use the dataset
for images, labels in train_dataset.take(1):
    print(images.shape, labels.shape)

# You can further preprocess your datasets as needed, for example, normalization
def preprocess_dataset(images, labels):
    images = tf.cast(images, tf.float32) / 255.0  # Normalize images to [0, 1]
    return images, labels

train_dataset = train_dataset.map(preprocess_dataset)
validation_dataset = validation_dataset.map(preprocess_dataset)

# Now, you can use train_dataset and validation_dataset in your model training
