import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np


# Sets global variables for batch size and image size
batch_size = 64
img_height = 180
img_width = 180

def createTrainingAndValidationData():
    global train_ds
    train_ds = tf.keras.utils.image_dataset_from_directory('A:\Anime_Dataset', validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width), batch_size=batch_size)

    global val_ds
    val_ds = tf.keras.utils.image_dataset_from_directory('A:\Anime_Dataset', validation_split=0.2, subset="validation", seed=123, image_size=(img_height, img_width), batch_size=batch_size)



createTrainingAndValidationData()

class_names = train_ds.class_names
print(class_names)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = len(class_names)



model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),


    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)

])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()

epochs = 15
model.fit(train_ds, validation_data=val_ds, epochs=epochs)


# model.save("newModel2.h5")
