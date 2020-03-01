import os
import getpass

from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.models import Model, Sequential
from keras.layers import Input, Dense, MaxPooling2D, Lambda, Activation, Conv2D, GlobalAveragePooling2D, BatchNormalization, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adamax
from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Parameters
DATASET_DIR = '/home/mcv/m5/datasets/MIT_split/'
BASE_PATH = '/home/grupo01/mcv/models'
NUM_CLASSES = 8
BATCH = 24
OPTIMIZER = 'Adamax'
IMAGE_SIZE = 64 
TRAIN_DATAGEN = ImageDataGenerator(preprocessing_function=preprocess_input,
    horizontal_flip=True,
    shear_range=2.0,
    featurewise_center=True)
TEST_DATAGEN = ImageDataGenerator(preprocessing_function=preprocess_input)

# Model
path = os.path.join(BASE_PATH, 'KerasModel_')
print('Building MLP model...\n')
    
model = Sequential()
model.add(Conv2D(32,
                kernel_size=(3, 3),
                activation='relu',
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                name='first_conv'))
model.add(MaxPooling2D(pool_size=(2, 2),
                    name='first_pool'))
model.add(BatchNormalization(axis=3,
                            momentum=0.999,
                            epsilon=1e-03))
model.add(Conv2D(64,
                kernel_size=(3, 3),
                activation='relu',
                name='second_conv'))
model.add(MaxPooling2D(pool_size=(2, 2),
                    name='second_pool'))
model.add(BatchNormalization(axis=3,
                            momentum=0.999,
                            epsilon=1e-03))
model.add(Conv2D(32,
                kernel_size=(1, 1),
                activation='relu',
                name='fourth_conv'))
model.add(GlobalAveragePooling2D())
model.add(Dense(64,
                activation='relu'))
model.add(Dense(8,
                activation='softmax'))

                
print(model.summary())


# Training model
model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=8, epsilon=1e-04, cooldown=0, min_lr=0)

train_generator = TRAIN_DATAGEN.flow_from_directory(
    DATASET_DIR+'/train',  # this is the target directory
    target_size=(IMAGE_SIZE, IMAGE_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
    batch_size=BATCH,
    classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

validation_generator = TEST_DATAGEN.flow_from_directory(
    DATASET_DIR+'/test',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH,
    classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    class_mode='categorical')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=1881 // BATCH,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=807 // BATCH,
    verbose=0,
    callbacks=[reduce_on_plateau])

print('Done!\n')
print('Saving the model\n')
model.save(path + 'model.h5')  # always save your weights after training or during training
print('Done!\n')


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(path + 'accuracy.jpg')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(path + 'loss.jpg')
plt.close()

num_parameters = model.count_params()
val_acc = history.history['val_acc'][-1]
ratio = val_acc*100000.0/float(num_parameters)
print('Optimizer: ', OPTIMIZER)
print('Ratio: ', ratio)

del model
del history
del train_generator
del validation_generator
