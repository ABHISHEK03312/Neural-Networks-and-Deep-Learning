#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.regularizers import l2
import tensorflow.keras.datasets.mnist as mnist

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


# In[2]:
batch_size = 128
num_classes = 10
no_epochs = 50


def train(para):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0


    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(625, activation='relu', kernel_regularizer=l2(para)),
        Dense(256, activation='relu', kernel_regularizer=l2(para)),
        Dense(100, activation='relu', kernel_regularizer=l2(para)),
        Dense(10, activation='softmax')])

    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=no_epochs,
                        verbose=2,
                        use_multiprocessing=False,
                        validation_data=(x_test, y_test))
    
    return history.history['val_loss'] #, history.history['val_accuracy']


def main():
    no_threads = mp.cpu_count()

    rates = [0.01, 0.001, 0.0001, 0.0]

    p = mp.Pool(processes = no_threads)
    loss = p.map(train, rates)

    plt.figure(1)
    for i in range(len(rates)):
        plt.plot(range(no_epochs), loss[i], label='beta = {}'.format(rates[i]))

    plt.xlabel('epochs')
    plt.ylabel('validation crossentropy')
    plt.legend()
    plt.savefig('./figures/5.4b_1.png')
    plt.show()
    
   # plt.figure(2)
   # for i in range(len(rates)):
    #    plt.plot(range(no_epochs), acc[i], label='beta = {}'.format(rates[i]))

    # plt.xlabel('epochs')
    # plt.ylabel('validation accuracy')
    # plt.legend()
    # plt.savefig('./figures/5.4b_2.png')
    # plt.show()

if __name__ == '__main__':
  main()




