#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:19:00 2018

@author: ayush
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 08:46:05 2018
for ix in range(0,10):
    plt.imshow(img)
    imshow(x_train_imgs[ix])
    plt.show()
@author: root
"""
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import SeparableConv2D
from keras import optimizers
from keras.utils import print_summary
import numpy as np
import matplotlib.pyplot as plt
#from skimage.io import imshow
import pandas as pd

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_test.shape
x_train.shape[3]
x_train[3]

x_train = x_train.reshape(50000,32,32,3).astype('float32')
x_test = x_test.reshape(10000,32,32,3).astype('float32')



x_train = x_train / 255
x_test = x_test / 255


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()


model.add(Conv2D(30, (5, 5), input_shape=( 32,32 , 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (3, 3)))

model.add(SeparableConv2D(32, (5,5)))
model.add(MaxPooling2D(pool_size = (3, 3)))

 
num_classes = 10
model.add(Flatten())



model.add(Dense(250, activation='relu'))
model.add(BatchNormalization(axis=1, momentum=0.9))
model.add(Dropout(rate=0.6))

model.add(Dense(150, activation='relu'))
model.add(BatchNormalization(axis=1, momentum=0.9))
model.add(Dropout(rate=0.6))

model.add(Dense(100, activation='relu'))
model.add(BatchNormalization(axis=1, momentum=0.9))
model.add(Dropout(rate=0.6))

model.add(Dense(300, activation='relu'))
model.add(BatchNormalization(axis=1, momentum=0.9))
model.add(Dropout(rate=0.6))


model.add(Dense(num_classes, activation='softmax'))


sgd = optimizers.SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
                            featurewise_center=True,
                            featurewise_std_normalization=True,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True)

datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
                    steps_per_epoch=len(x_train) / 32, epochs=10)




history = model.fit(x_train, y_train, epochs= 10, batch_size=100, validation_split = 0.3)
scores = model.evaluate(x_test, y_test, verbose = 0 )

historydf = pd.DataFrame(history.history,index= history.epoch)
historydf.plot()
#historydf.plot(ylim = (0,1))

plt.title("Test Accuracy: {:3.1f} % ".format(scores[1]*100),fontsize = 15)


x_train_imgs = np.zeros([50000,32,32,3])
for i in range(x_train.shape[0]):
    img = x_train[i,:].reshape([32,32,3])
    x_train_imgs[i] = img

print ( scores )
print_summary(model, line_length=None, positions=None, print_fn=None)



model.save('model.h5')











#
######################################################################################
#from keras.models import load_model
#import cv2
#import numpy as np
#
#model = load_model('model/model.h5')
#
#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])
#
#img = cv2.imread('images/pic03.jpg')
#img = cv2.resize(img,(32,32))
#img = np.reshape(img,[1,32,32,3])
#
#classes = model.predict_classes(img)
#
#print(classes)
#
##
#
#
#
#
#










