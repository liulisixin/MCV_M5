from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Flatten
from keras.layers import (
        Dense, 
        Reshape, 
        GlobalAveragePooling2D, 
        Conv2D, SeparableConv2D, 
        BatchNormalization, 
        AveragePooling2D, 
        MaxPooling2D, 
        Flatten, 
        Activation
)
from keras import backend as K
from keras.utils import plot_model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


train_data_dir='../../MIT_split/train'
val_data_dir='../../MIT_split/test'
test_data_dir='../../MIT_split/test'
img_width = 256
img_height=256
batch_size=32
number_of_epoch=200
validation_samples=807

# Create the well balanced model
model = Sequential()


model.add(Conv2D(48, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=(img_width, img_height, 3), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(48, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(SeparableConv2D(48, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(SeparableConv2D(48, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(AveragePooling2D(pool_size=(2,2)))


model.add(SeparableConv2D(62, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(62, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(AveragePooling2D(pool_size=(2,2)))


model.add(SeparableConv2D(62, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(SeparableConv2D(62, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(AveragePooling2D(pool_size=(2,2)))


model.add(SeparableConv2D(62, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(SeparableConv2D(62, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=8, activation='softmax'))

optimizer = optimizers.Adam()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

test_datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

train_generator = test_datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

history=model.fit_generator(train_generator,
        steps_per_epoch=(int(400//batch_size)+1),
        nb_epoch=number_of_epoch,
        validation_data=validation_generator,
        validation_steps= (int(validation_samples//batch_size)+1))


result = model.evaluate_generator(test_generator, val_samples=validation_samples)
print(result)

# list all data in history
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy.jpg')
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss.jpg')