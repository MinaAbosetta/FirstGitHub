# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:10:05 2020


Sarcasm Classification


@author: MinaAbosetta
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
import json
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# %% defin the call back function
###########################
class myCallBack(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>=0.99):
          print("\nReached 99% accuracy so cancelling training!")
          self.model.stop_training = True

callbacks= myCallBack()


#%% import input data and initial variables 

corpus  = []
lable   = []
max_len = 0

with open("sarcasm.json",'r') as dataset:
    data= json.load(dataset)
    
       
for item in data:
    corpus.append(item['headline'])
    lable.append(item['is_sarcastic'])
    if len(item['headline'].split()) > max_len:
        max_len= len(item['headline'].split())  
        
corpus = np.array(corpus)
lable  = np.array(lable)

# Shaffling the data 
corpus,lable= shuffle(corpus,lable, random_state= 0)

# Spliting the Data into train and test sets  
spliting     = 24000
Train_corpus = corpus[:spliting]
Train_lable  = lable[:spliting]

Test_corpus  = corpus[spliting:]
Test_lable   = lable[spliting:]


#%% text Tokenizing and padding

tokenizer       = Tokenizer(num_words =10000, oov_token = '<oov>')
tokenizer.fit_on_texts(Train_corpus)


# tokenizing and padding the training set
train_sequences = tokenizer.texts_to_sequences(Train_corpus)
train_padding   = pad_sequences(train_sequences, padding= 'post', maxlen= max_len)

# tokenizing and padding the test set
test_sequences  = tokenizer.texts_to_sequences(Test_corpus)
test_padding    = pad_sequences(test_sequences, padding= 'post', maxlen= max_len)


#%% Classification Model definition
# It is a sequential model:
    # layer 1 :  Embedding layer
    # layer 2 :  Bidirectional LSTM layer
    # layer 3 :  Bidirectional LSTM layer
    # layer 4-6: fully connected layers (Dense layers)
    
model= tf.keras.models.Sequential([tf.keras.layers.Embedding(10000, 32, input_length= max_len),
                                   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True,activation= 'tanh')),
                                   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,activation= 'tanh')),
                                   tf.keras.layers.Dense(60, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.2)),
                                   tf.keras.layers.Dense(30, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.2)),
                                   tf.keras.layers.Dense(1, activation= 'sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l=0.2))
                                    ])


model.summary() # Print The model summary


# Compile the model with 'Adam' as an optimizer                                   
model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['acc']) 

# Fitting the model
history= model.fit (train_padding,Train_lable,epochs= 100, callbacks=[callbacks], validation_data= (test_padding,Test_lable))     
  

#%% ploting the result 

# The training/test accuracy at each iteration
plt.figure(1)
plt.plot(history.epoch, history.history['acc'],'r-*',label='Training Accuracy')
plt.plot(history.epoch, history.history['val_acc'],'b-*',label='Testing Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.grid()
plt.legend()
plt.show()

# The training/test loss at each iteration
plt.figure(2)
plt.plot(history.epoch, history.history['loss'],'r*-',label='Training Loss')
plt.plot(history.epoch, history.history['val_loss'],'b-*',label='Testing Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid()
plt.show()

