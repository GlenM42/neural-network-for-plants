import tensorflow as tf
import keras
from keras import layers
import os
from PIL import Image
import keras_tuner as kt

img_w, img_h = 256, 256


# Function to check and resave images for integrity
def check_and_resave_images(image_dir):
    for subdir, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as img:
                        img.save(file_path)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")


# Prompt for image check
user_input = input("Do you want to recheck all the images? (N/y) \n").lower()
if user_input == 'y':
    image_dir = 'images'
    check_and_resave_images(image_dir)
else:
    print("Skipping image check.\n")


# Model builder function for Keras Tuner
def model_builder(hp):
    model = keras.Sequential()

    # Hyperparameters for data augmentation
    hp_rotation = hp.Float('rotation_factor', min_value=0.0, max_value=0.5, step=0.1)
    data_augmentation = keras.Sequential([
        layers.RandomRotation(hp_rotation),
    ])
    model.add(data_augmentation)

    # Convolutional and pooling layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_w, img_h, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Hyperparameters for dropout rate
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    model.add(layers.Dropout(hp_dropout))

    # Final layers
    model.add(layers.Flatten())
    hp_l2 = hp.Float('l2_regularization', min_value=1e-5, max_value=1e-2, sampling='LOG')
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(hp_l2)))
    model.add(layers.Dense(4, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Prepare datasets
(train_set, validation_set) = keras.utils.image_dataset_from_directory(
    directory="images",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(img_w, img_h),
    shuffle=True,
    seed=123,
    validation_split=0.2,  # Adjust the split ratio as needed
    subset="training",
    interpolation="bilinear",
), keras.utils.image_dataset_from_directory(
    directory="images",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(img_w, img_h),
    shuffle=True,
    seed=123,
    validation_split=0.2,  # Adjust the split ratio as needed
    subset="validation",
    interpolation="bilinear",
)

# Configure the tuner
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     directory='images',
                     project_name='neural-network-for-plants')

# Perform hyperparameter search
tuner.search(train_set, epochs=10, validation_data=validation_set)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal rotation factor is {best_hps.get('rotation_factor')},
the optimal dropout rate is {best_hps.get('dropout')}, and
the optimal L2 regularization factor is {best_hps.get('l2_regularization')}.
""")

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_set, epochs=10, validation_data=validation_set)
