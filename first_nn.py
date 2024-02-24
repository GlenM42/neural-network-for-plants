# Phillip Waul
# 2/20/24
# This is a barebones CNN where we just have it go over our dataset with little to no real changes.
# first neural network with keras tutorial

import tensorflow as tf
import keras
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import Image
import matplotlib.pyplot as plt
import os
from keras import models
from keras import layers
from keras.datasets import mnist
img_w, img_h = 256, 256
#image_dir = 'images'  # Path to your images directory
#for subdir, dirs, files in os.walk(image_dir):
#    for file in files:
#        extension = os.path.splitext(file)[1]
#        if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
#            file_path = os.path.join(subdir, file)
#            try:
#                with Image.open(file_path) as img:
#                    img.save(f"{file_path}.{extension}")  # Attempt to re-save the image
#                   os.replace(f"{file_path}.{extension}", file_path)  # Replace original with re-saved image
#            except Exception as e:
#                print(f"Error processing file {file_path}: {e}")

# Load the dataset from the directory
(train_set, test_set) = keras.utils.image_dataset_from_directory(
    directory="images",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(img_w, img_h),
    shuffle=True,
    seed=123,  # Seed for shuffling and transformations
    validation_split=0.5,  # Adjust the split ratio as needed
    subset="both",
    interpolation="bilinear",
)


#this function helps seperate out the list of images and list of labels in each set of data.
def extract_images_labels(image, label):
    return image, label

# Apply the extract_images_labels function to the datasets using map
train_set = train_set.map(extract_images_labels)
test_set = train_set.map(extract_images_labels)
# Now, you can use the `unbatch` method to separate images and labels
train_images_set = train_set.map(lambda image, label: image)
train_labels_set = train_set.map(lambda image, label: label)
test_images_set = train_set.map(lambda image, label: image)
test_labels_set = train_set.map(lambda image, label: label)
#The labels are still a Map datatype, but arrays are easier to work with so let's use those.
train_images_array = tf.concat(list(train_images_set.as_numpy_iterator()), axis=0)
train_labels_array = tf.concat(list(train_labels_set.as_numpy_iterator()), axis=0)
test_images_array = tf.concat(list(train_images_set.as_numpy_iterator()), axis=0)
test_labels_array = tf.concat(list(train_labels_set.as_numpy_iterator()), axis=0)

# Now, you can use the arrays in the model training
print(test_images_array.dtype)
print(test_labels_array.shape)
print(test_labels_array.dtype)

# The example I based this off of used a 5:1 ratio, so that's what I'm trying.
partial_train_images = train_images_array[84:]
train_images_val = train_images_array[:84]
train_labels_val = train_labels_array[:84]
partial_train_labels=train_labels_array[84:]

# Making sure that the iamges are using real numbers rather than bytes. This is better for convolution.
# Some guides will also reshape the data at this time. 
# Luckily, keras.utils.image_dataset_from_directory does that for us
#train_images_val = train_images_val.astype('float32')/255
#artial_train_images=partial_train_images.astype('float32')/255
#test_images_array = test_images_array.astype('float32') / 255



#NEURAL NETWORK
# Conv2D layers: This moves a "filter" over the image step by step. It does vector multiplication to determine if the image matches a "template" that the filter learns through testing.
#               The result is kind of like a smaller image with fewer pixels, but each pixel has more meaning.
#       format: layers.Conv2D(number of filters, (filter width, filter height), activation, input shape)
#               The input shape is based on the image data arrays. They are in the format: [number of samples, img width, img height, color channels]
#               We leave out the number of samples since we are putting each sample through this process
#               Other arguments could initialize a "template" for the filter, initialize a bias, or put padding around the image so the filter can go a little off the edge.
#               More on activations later.
# MaxPooling2D layers: MaxPooling will take the max value in a section of the convolution and use that value to represent the whole section.
#                       Will explain more in the README, but the idea is that it summarises a region so there are fewer things the AI needs to learn to achieve results
#       format: layers.MaxPooling2D((width, length))
#                this means width and length of the square that will get summarized.
# 
# Flatten layer: This just reshapes our data a bit so that we can get an answer from it. Basically it's not an image anymore, now it's parameters for a normal AI.
# 
# Dense layers: These are regular neural network layers. They multiply parameters together at special weights to get an answer.
#               The network will do math to figure out which weights will give the best answer!
#       format: layers.Dense(number of nodes, activation function)
#
# Activation functions: Activation functions are just the specific math equation that the network uses to determine the output of all the multiplication.
#               ReLU: Rectified linear unit - basically if the math gives a negative number, it is treated as a zero. If the math gives a positive number, that number is the result.
#                       This makes it work so that It doesn't punish being incorrect, but it rewards being correct.
#               Softmax: Basically it normalizes the output and turns it into a probability of each different classification being correct.
#                       For example, if the network thinks the image is female, it may output [.8, .1, .05, .05]. In this scenario, let's assume the first output is for female, then the next for male, both, then sterile.
#                       So, the output of the softmax says it thinks there's an 80% chance the image is female. And a 10% chance of male, .05 of both, and .05 of sterile.
#                       For this, we need the number of nodes to be the same as our number of outputs


cnn = models.Sequential()
cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_w, img_h, 3)))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn.add(layers.Flatten())
cnn.add(layers.Dense(64, activation='relu'))
cnn.add(layers.Dense(4, activation='softmax'))
print(cnn.summary())
cnn.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
history=cnn.fit(partial_train_images,
                    partial_train_labels, epochs=5, batch_size=64,
                    validation_data=(train_images_val, train_labels_val))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
# "bo" is for "blue dot"
#plt.figure(1)
fig,axis = plt.subplots(nrows=2, ncols=1)
fig.tight_layout()
plt.subplot(211)
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.xticks([])
plt.yticks([])
plt.subplot(212)
plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
#test_loss, test_acc = cnn.evaluate(test_images_array, test_labels_array)
#print('test_acc:', test_acc)
# At this point we have a neural network that runs. This took less than a minute to run.
# This still needs a little bit of work since the tests did not run correctly. It is cool to see the learning accuracy go up with each epoch...
# But it could theoretically just be learning these images. It is not truly a success unless it is successful on test data as well
# Currently the loss is either... atrocious or malfunctioning. Hopefully the latter. Needs a bit more work.
