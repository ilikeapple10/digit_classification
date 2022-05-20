#import libraries
from pickletools import pylong
from pyexpat import model
from matplotlib.cbook import flatten
import tensorflow as tf
from keras.datasets import mnist
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import random

#download the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path="mnist.npz")

#create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28, 1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (2, 2), activation = "relu"))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (2, 2), activation = "relu"))

#classifies the image
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dense(10, activation = "softmax"))
model.summary()

#prepare the data for training
x_train = train_images.reshape(train_images.shape[0], 28, 28, 1)
x_test = test_images.reshape(test_images.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,train_labels, epochs=1)
test_loss, test_acc = model.evaluate(x_test, test_labels)

#output the prediction
while True:
    fig = pyplot.figure()
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

    image_index = random.randrange(0, 9991)

    for i in range (9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(x_test[image_index + i], cmap = pyplot.get_cmap('gray'))
        predict = x_test[i].reshape(28,28)
        pred = model.predict(x_test[image_index + i].reshape(1, 28, 28, 1))

    for i in range (9):
        ax = pyplot.subplot(330 + 1 + i)
        pred = model.predict(x_test[image_index + i].reshape(1, 28, 28, 1))
        ax.title.set_text(pred.argmax())
    pyplot.show()
