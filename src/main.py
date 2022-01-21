# %% Imports
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib

# %% check tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

# %% var
data_path = Path('data/raw')
train_path = data_path.joinpath('asl_alphabet_train/asl_alphabet_train')
train_path = data_path.joinpath('asl-own')
# test_path = data_path.joinpath('asl-alphabet-test')
models_path = Path('models')

# %% generators

data_generator = ImageDataGenerator(
    # rotation_range=10,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1 / 255.0,
    brightness_range=(0.75, 1.3),
    zoom_range=0.2,
    validation_split=0.1
)
valid_generator = ImageDataGenerator(
    rescale=1 / 255.0,
    validation_split=0.1
)

train_gen = data_generator.flow_from_directory(
    directory=train_path,
    target_size=(200, 200),
    color_mode="rgb",
    batch_size=64,
    class_mode="sparse",
    seed=2022,
    subset="training"
)
valid_gen = valid_generator.flow_from_directory(
    directory=train_path,
    target_size=(200, 200),
    color_mode="rgb",
    batch_size=64,
    class_mode="sparse",
    seed=2022,
    subset="validation"
)
classes = list(train_gen.class_indices.keys())
print(classes)

def decode_class(cls, one_hot):
    return cls[np.argmax(one_hot)]


# %% test generators
for i in range(10):
    image, label = next(train_gen)
    print(decode_class(classes, label))
    image = image[0] * 255.0
    image = image.astype('uint8')
    image = np.squeeze(image)
    cv2.imshow('i', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# %% Build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(200, 200, 3)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(tf.keras.layers.MaxPooling2D(3, 3))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(tf.keras.layers.MaxPooling2D(3, 3))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(tf.keras.layers.MaxPooling2D(3, 3))

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(1500, activation="relu"))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(26, activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.summary()

# %% Fit
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=models_path.joinpath('{epoch:02d}-{val_loss:.2f}.hdf5'),
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=False)

history = model.fit(train_gen,
                    epochs=999,
                    validation_data=valid_gen,
                    callbacks=[model_checkpoint_callback, tf.keras.callbacks.EarlyStopping(patience=2)],
                    workers=20)

# %% Plot
training_loss = history.history['loss']
test_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epoch_count = range(1, len(training_loss) + 1)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_count, training_loss)
plt.plot(epoch_count, test_loss)
plt.legend(['Train Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(epoch_count, train_acc)
plt.plot(epoch_count, val_acc)
plt.legend(['Train Accuracy', 'Validation Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
