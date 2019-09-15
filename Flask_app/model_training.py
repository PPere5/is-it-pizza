# simplified interface for building models
import keras
# our handwritten character labeled dataset
from keras.datasets import mnist
# because our models are simple
from keras.models import Sequential
# dense means fully connected layers, dropout is a technique to improve convergence, flatten to reshape our matrices for feeding
# into respective layers
from keras.layers import Dense, Dropout, Flatten
# for convolution (images) and pooling is a technique to help choose the most relevant features in an image
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

import load_pizza

# mini batch computer's RAM cannot handle large batches
batch_size = 1
# pizza or not pizza boolean
num_classes = 2
# very short training time
epochs = 5


# input image dimensions
img_rows, img_cols = 200, 200
input_shape = (img_rows, img_cols, 3)

test_ratio = 0.3
seed = 42
n_per_folder = 80

# the data downloaded, shuffled and split between train and test sets
# if only all datasets were this easy to import and format

(x_train, y_train), (x_test, y_test) = load_pizza.load_pizza_data(img_rows, img_cols, test_ratio, seed, n_per_folder)

# this assumes our data format
# For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while
# "channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).

# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# more reshaping

x_train = np.array([im.astype('float32') for im in x_train])
x_test = np.array([im.astype('float32') for im in x_test])

x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train)
print(y_test)

# build our model
model = Sequential()
# convolutional layer with rectified linear unit activation
model.add(Conv2D(204, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# split image into smaller subsets of images
model.add(Conv2D(408, (3, 3), activation='relu'))
# pooling to select relevant part of image
model.add(MaxPooling2D(pool_size=(2, 2)))
# use dropout quite a bit to prevent overfitting
model.add(Dropout(0.3))
# flatten since too many dimensions, we only want a classification output
model.add(Flatten())
# fully connected to get all relevant data, interleaved with droupout to let go some reinforcing perceptrons
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# output a softmax to squash the matrix into output probabilities
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# train
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
 # how well did it do?
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Save the model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
