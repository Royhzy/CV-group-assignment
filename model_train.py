from __future__ import print_function
import matplotlib.pyplot as plt
import random
import numpy as np
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import utils as np_utils
from skimage.util import img_as_float
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import skimage
from data_input import extract_data, resize_with_pad, IMAGE_SIZE, read_file
import tensorflow as tf
from tkinter import messagebox

faceData_file_path = './faceData/'


class Dataset(object):

    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.Y_train = None
        self.Y_valid = None
        self.Y_test = None

    def read(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=3):
        # images, labels = extract_data('./faceData/')
        images, labels, counter = read_file(faceData_file_path)
        nb_classes = counter     # Number of folders
        print('nb_classes',nb_classes)
        # print('images',images)
        #print('labels',labels)

        labels = np.reshape(labels, [-1])
        # numpy.reshape
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3,
                                                            random_state=random.randint(0, 100))
        X_valid, X_test, y_valid, y_test = train_test_split(images, labels, test_size=0.5,
                                                            random_state=random.randint(0, 100))
        if K.image_data_format() == "channels_first":
            X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
            X_valid = X_valid.reshape(X_valid.shape[0], 3, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
            X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 3)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

        # the data, shuffled and split between train and test sets
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices (one-hot)
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_valid = np_utils.to_categorical(y_valid, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        X_train = X_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_valid /= 255
        X_test /= 255

        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_valid = Y_valid
        self.Y_test = Y_test


class Model(object):
    FILE_PATH = './model/model.h5'

    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=3):
        images, labels, counter = read_file(faceData_file_path)
        nb_classes = counter

        self.model = Sequential()
        print(dataset.X_train.shape[1:])
        # Convolutional layer
        self.model.add(Convolution2D(32, 3, 3, padding='same', input_shape=dataset.X_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        # Max pool
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # Dropout layer anti overfitting
        self.model.add(Dropout(0.25))
        # Second convolution
        self.model.add(Convolution2D(64, 3, 3, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3, padding='same'))
        self.model.add(Activation('relu'))
        # Second max pool
        self.model.add(MaxPooling2D(pool_size=(2, 2,), padding='same'))
        self.model.add(Dropout(0.25))
        #The main purpose of the Flatten layer is to transform the multidimensional matrix above into a one-dimensional matrix for use by the fully connected layer, avoiding overfitting
        self.model.add(Flatten())
        #Fully connected layer as intermediate layer
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        #Fully connected layer as intermediate layer
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))  # The activation function of this layer is no longer using ReLU, mainly because it is used to determine the result

        self.model.summary()

    def train(self, dataset, batch_size=32, nb_epoch=2000, data_augmentation=False):
        # let's train the model using SGD + momentum (how original).
        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        if not data_augmentation:
            print('Not using data augmentation.')
            H = self.model.fit(dataset.X_train, dataset.Y_train,
                               batch_size=batch_size,
                               epochs=nb_epoch,
                               validation_data=(dataset.X_valid, dataset.Y_valid),
                               shuffle=True)
        else:
            print('Using real-time data augmentation.')

            # this will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the 【
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(dataset.X_train)

            # fit the model on the batches generated by datagen.flow()
            H = self.model.fit_generator(datagen.flow(dataset.X_train, dataset.Y_train,
                                                      batch_size=batch_size),
                                         steps_per_epoch=dataset.X_train.shape[0],
                                         epochs=nb_epoch,
                                         validation_data=(dataset.X_valid, dataset.Y_valid))
        N = np.arange(0, nb_epoch)

        #Draw a training diagram
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["accuracy"], label="train_acc")
        plt.plot(N, H.history["val_accuracy"], label="val_acc")
        plt.title("Training and Validation Accuracy (Simple NN)")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()
        print("ok")
        plt.savefig('./Picture/simple_nn_plot_acc.png')

        plt.figure()
        plt.plot(N, H.history["loss"], label="train_loss")
        plt.plot(N, H.history["val_loss"], label="val_loss")
        plt.title("Training and Validation Loss (Simple NN)")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('./Picture/simple_nn_plot_loss.png')

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    def predict(self, image):

        image = resize_with_pad(image)
        print("2")
        image = tf.reshape(image, [1, 64, 64, 3])
        print(image.shape)
        image = img_as_float(image)
        # print(f"image",image)
        image /= 255  # Convert the image matrix to a range of 0-1

        result = self.model.predict(image)
        max_index = np.argmax(result,axis=1)  # Find the one with the highest probability

        return max_index[0]


    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))


def start():
    dataset = Dataset()
    dataset.read()

    model = Model()
    model.build_model(dataset)
    model.train(dataset, nb_epoch=40000)
    model.save('myModelName.h5')  #save model
    model = Model()
    model.load()
    model.evaluate(dataset)
    messagebox.showinfo("model train Finished","Successfully train！")  # successful prompt
