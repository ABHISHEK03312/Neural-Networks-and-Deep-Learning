#!/usr/bin/env python
# coding: utf-8

# In[1]:

## Example 4.5b: Experiments with batch sizes on MNIST data

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.datasets.mnist as mnist

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp


# In[2]:

num_classes = 10
no_epochs = 30

class time_for_batch(Callback):
    def on_train_begin(self, logs={}):
        self.times=[]
    def on_train_batch_begin(self, batch, logs={}):
        self.starttime = time.time()
    def on_train_batch_end(self, batch, logs={}):
        self.times.append(time.time()-self.starttime)
        
class time_for_epoch(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
    
def my_train(batch_size):
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(625, activation='relu'),
        Dense(100, activation='relu'),
        Dense(10, activation='softmax')])

    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    tb = time_for_batch()
    te = time_for_epoch()
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=no_epochs,
                        verbose=2,
                        use_multiprocessing=False,
                        callbacks = [tb, te],
                        validation_data=(x_test, y_test))

    paras = np.array([min(history.history['val_loss']), 
                      max(history.history['val_accuracy']), 
                      sum(tb.times)/len(tb.times), 
                      sum(te.times)/len(te.times)])
    
    return paras


def main():
  batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512]

  no_threads = mp.cpu_count()
  p = mp.Pool(processes = no_threads)
  paras = p.map(my_train, batch_sizes)

  paras = np.array(paras)
  entropy, accuracy, time_batch, time_epoch = paras[:,0], paras[:,1], paras[:, 2], paras[:, 3]

  plt.figure(1)
  plt.plot(range(len(batch_sizes)), entropy)
  plt.xticks(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('batch size')
  plt.ylabel('entropy')
  plt.title('test entropy vs. batch size')
  plt.savefig('./figures/4.5b_1.png')

  plt.figure(2)
  plt.plot(range(len(batch_sizes)), accuracy)
  plt.xticks(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('batch size')
  plt.ylabel('test accuracy')
  plt.title('test accuracy vs. batch size')
  plt.savefig('./figures/4.5b_2.png')

  plt.figure(3)
  plt.plot(range(len(batch_sizes)), time_batch)
  plt.xticks(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('batch size')
  plt.ylabel('time (ms)')
  plt.title('time for a batch vs. batch size')
  plt.savefig('./figures/4.5b_3.png')
  
  plt.figure(4)
  plt.plot(range(len(batch_sizes)), time_epoch)
  plt.xticks(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('batch size')
  plt.ylabel('time (ms)')
  plt.title('time for an epoch vs. batch size')
  plt.savefig('./figures/4.5b_4.png')
 
#  plt.show()

if __name__ == '__main__':
  main()






