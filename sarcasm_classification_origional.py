# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:10:05 2020

@author: minaa
"""
#%% delet all variable before start
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
if __name__ == "__main__":
    clear_all()
#%% import libraries

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# %% defin the call back
###########################
class myCallBack(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>=0.99):
          print("\nReached 99% accuracy so cancelling training!")
          self.model.stop_training = True

callbacks= myCallBack()


#%%

corpus=[]
lable=[]
max_len= 0

with open("sarcasm.json",'r') as dataset:
    data= json.load(dataset)
    
    
    
for item in data:
    corpus.append(item['headline'])
    lable.append(item['is_sarcastic'])
    if len(item['headline'].split()) > max_len:
        max_len= len(item['headline'].split())  
        
corpus = np.array(corpus)
lable= np.array(lable)
unique, counts = np.unique(lable, return_counts=True)
dict(zip(unique, counts))

# Shaffling the data 
corpus,lable= shuffle(corpus,lable, random_state= 0)

# Splitiing the Data to train and test sets  
spliting= 24000
Train_corpus= corpus[:spliting]
Train_lable= lable[:spliting]

Test_corpus= corpus[spliting:]
Test_lable= lable[spliting:]


tokenizer= Tokenizer(num_words =10000, oov_token = '<oov>')
tokenizer.fit_on_texts(Train_corpus)
train_sequences= tokenizer.texts_to_sequences(Train_corpus)
train_padding= pad_sequences(train_sequences, padding= 'post', maxlen= max_len)


test_sequences= tokenizer.texts_to_sequences(Test_corpus)
test_padding= pad_sequences(test_sequences, padding= 'post', maxlen= max_len)
#%% Model 1 
model= tf.keras.models.Sequential([tf.keras.layers.Embedding(10000, 32, input_length= max_len),
                                   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True,activation= 'tanh')),
                                   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,activation= 'tanh')),
                                   tf.keras.layers.Dense(60, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.2)),
                                   tf.keras.layers.Dense(30, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.2)),
                                   tf.keras.layers.Dense(1, activation= 'sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l=0.2))
                                    ])



# lr_callbacks= tf.keras.callbacks.LearningRateScheduler(
#         lambda epoch: 1e-8 * 10**(epoch / 20))

# optimizer= tf.keras.optimizers.Adam(lr= 1e-4)      
                                  
model.summary()
                                   
model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['acc'])
history= model.fit (train_padding,Train_lable,epochs= 30, callbacks=[callbacks], validation_data= (test_padding,Test_lable))     
# history= model.fit (train_padding,Train_lable, epochs= 160, callbacks=[lr_callbacks])     
# history= model.fit (train_padding,Train_lable, epochs= 30, validation_data= (test_padding,Test_lable))     

#%% ploting 
plt.figure(0)
plt.semilogx(history.history["lr"],history.history['loss'])
plt.grid()
plt.show()


plt.figure(1)
plt.plot(history.epoch, history.history['acc'],'r-*',label='Training Accuracy')
plt.plot(history.epoch, history.history['val_acc'],'b-*',label='Testing Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.grid()
plt.legend()
plt.show()


plt.figure(2)
plt.plot(history.epoch, history.history['loss'],'r*-',label='Training Loss')
plt.plot(history.epoch, history.history['val_loss'],'b-*',label='Testing Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid()
plt.show()

