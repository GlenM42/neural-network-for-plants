import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras import models
from keras import layers
import os
from PIL import Image

img_w, img_h = 256, 256


# this function helps separate out the list of images and list of labels in each set of data.
def extract_images_labels(image, label):
    return image, label


def check_and_resave_images(image_dir):
    """
    Check all JPEG images in the specified directory and its subdirectories,
    and attempt to re-save them. This is useful for verifying image integrity.
    """
    for subdir, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as img:
                        # Re-save the image without changing its extension
                        img.save(file_path)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")


# Prompt the user to decide whether to check and re-save images
user_input = input("Do you want to recheck all the images? (N/y) \n").lower()
if user_input == 'y':
    image_dir = 'images'  # Path to your images directory
    check_and_resave_images(image_dir)
else:
    print("Skipping image check.\n")

# Highlight: Implementing data augmentation to reduce overfitting
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

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

# Apply the extract_images_labels function to the datasets using map
train_set = train_set.map(extract_images_labels)
test_set = train_set.map(extract_images_labels)
# Now, you can use the `unbatch` method to separate images and labels
train_images_set = train_set.map(lambda image, label: image)
train_labels_set = train_set.map(lambda image, label: label)
test_images_set = train_set.map(lambda image, label: image)
test_labels_set = train_set.map(lambda image, label: label)
# The labels are still a Map datatype, but arrays are easier to work with so let's use those.
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
partial_train_labels = train_labels_array[84:]

cnn = models.Sequential([
    # Highlight: Adding the data augmentation layer to the model
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_w, img_h, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # Highlight: Adding dropout to reduce overfitting
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax'),
])
cnn.build(input_shape=(None, img_w, img_h, 3))
print(cnn.summary())

# Highlight: Adding L2 regularization to the dense layer
regularizer = keras.regularizers.l2(0.001)  # You can adjust the L2 regularization factor
for layer in cnn.layers:
    if hasattr(layer, 'kernel_regularizer'):
        setattr(layer, 'kernel_regularizer', regularizer)

cnn.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
history = cnn.fit(partial_train_images,
                  partial_train_labels, epochs=5, batch_size=64,
                  validation_data=(train_images_val, train_labels_val))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
# "bo" is for "blue dot"
# plt.figure(1)
fig, axis = plt.subplots(nrows=2, ncols=1)
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
# test_loss, test_acc = cnn.evaluate(test_images_array, test_labels_array) print('test_acc:', test_acc)
