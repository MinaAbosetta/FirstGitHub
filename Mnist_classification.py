# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:48:26 2020

@author: minaa
"""


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %% defin the call back
###########################
class myCallBack(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>=0.99):
          print("\nReached 99% accuracy so cancelling training!")
          self.model.stop_training = True

callbacks= myCallBack()

#%% load the data set and preprocessing 
##########################################

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train= x_train/255.0
x_test= x_test/255.0

x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)

#%% model definition (model_structure, compiling, fitting)
##########################################################

model= tf.keras.models.Sequential ([tf.keras.layers.Conv2D(64,(3,3), activation= 'relu', input_shape= (28,28,1)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(128,(3,3), activation= 'relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(512,(3,3), activation= 'relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64,activation= 'relu'),
                                    tf.keras.layers.Dense(10, activation= 'softmax')])
                             
model.compile(optimizer= 'sgd', metrics= ['acc'], loss='sparse_categorical_crossentropy')
history= model.fit (x_train,y_train, epochs= 100, validation_data= (x_test, y_test), callbacks= [callbacks])
# history= model.fit (x_train,y_train, epochs= 100, validation_data= (x_test, y_test))



# %% plotting the training accuracy and validation accuracy per epoch
##########################################################
plt.figure(0)
plt.plot (history.epoch, history.history ['acc'])
plt.plot (history.epoch, history.history ['val_acc'])


plt.figure(1)
plt.plot (history.epoch, history.history ['loss'])
plt.plot (history.epoch, history.history ['val_loss'])

# %% Confusion matrix of the tset set
##########################################################
matrix = tf.math.confusion_matrix(y_test, model.predict_classes(x_test)).numpy()
matrix = np.round(matrix/matrix.sum(axis=1), decimals= 2)

figure = plt.figure(1)
sns.heatmap(matrix, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
