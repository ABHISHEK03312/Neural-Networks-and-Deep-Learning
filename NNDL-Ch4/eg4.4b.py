#!/usr/bin/env python
# coding: utf-8

# In[1]:
## Example 4.4b: Investiging depth of DNN on Boston Housing data

import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import pylab as plt
from sklearn import preprocessing
import multiprocessing as mp

import os
if not os.path.isdir('figures'):
    os.makedirs('figures')


# In[2]:
learning_rate = 0.001

no_epochs = 200
batch_size = 32

seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)


def my_train(no_layers):
    
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    model = tf.keras.Sequential()
    for l in range(no_layers):
        model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='sgd',
                  loss='mse')

    history = model.fit(x_train, y_train, 
                        epochs=no_epochs, 
                        batch_size=batch_size, 
                        verbose = 2, 
                        use_multiprocessing=True,
                        validation_data=(x_test, y_test))

    return history.history['val_loss']



def main():

  hidden_layers = [1, 2, 3, 4, 5, 6]

  no_threads = mp.cpu_count()
  p = mp.Pool(processes = no_threads)
  cost = p.map(my_train, hidden_layers)

  # plot learning curves
  plt.figure(1)

  min_cost = []
  for l in range(len(hidden_layers)):
    plt.plot(range(no_epochs), cost[l], label = 'layers = {}'.format(hidden_layers[l] + 2))
    min_cost.append(min(cost[l]))

  plt.xlabel('epochs')
  plt.ylabel('mean square error')
  plt.title('GD learning')
  plt.legend()
  plt.savefig('figures/4.4b_1.png')

  
  plt.figure(2)
  plt.plot(hidden_layers, min_cost)
  plt.xlabel('the depth')
  plt.ylabel('test error')
  plt.title('test error vs.the depth of DNN')
  plt.savefig('figures/4.4b_2.png')

#  plt.show()


if __name__ == '__main__':
  main()






