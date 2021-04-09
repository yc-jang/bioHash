  
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt

from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

import tensorflow as tf
mnist  = tf.keras.datasets.mnist
"""[mnist load data]
    train_x : (60000, 28, 28)
    train_y : (60000, )
    test_x  : (10000, 28, 28)
    test_y  : (10000, )

Returns:
    [torch.uint8]: [reshape data]
    train_x : ([60000, 784])
    test_x  : ([10000, 784])
    train_y : ([60000])
    test_y  : ([10000])

"""
(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = mnist.load_data() #load data
mnist_train_x, mnist_test_x = torch.tensor(mnist_train_x.reshape(-1, 28*28, order="F")/255, dtype=torch.float), torch.tensor(mnist_test_x.reshape(-1, 28*28, order="F")/255, dtype=torch.float)
mnist_train_y, mnist_test_y = torch.tensor(mnist_train_y), torch.tensor(mnist_test_y)

cifar10  = tf.keras.datasets.cifar10
"""[summary]
    cifar10_train_x_raw : (50000, 32, 32, 3)
    cifar10_train_y_raw : (50000, 1), min-max [0-9]
    cifar10_test_x_raw  : (10000, 32, 32, 3)
    cifar10_test_y_raw  : (10000, 1), min-max [0-9]
Returns:
    [type]: [description]
"""
(cifar10_train_x_raw, cifar10_train_y_raw), (cifar10_test_x_raw, cifar10_test_y_raw) = cifar10.load_data() #load data
cifar10_train_x, cifar10_test_x = torch.tensor(cifar10_train_x_raw.reshape(-1, 32*32*3, order="F")/255, dtype=torch.float), torch.tensor(cifar10_test_x_raw.reshape(-1, 32*32*3, order="F")/255, dtype=torch.float)
cifar10_train_y, cifar10_test_y = torch.tensor(cifar10_train_y_raw), torch.tensor(cifar10_test_y_raw)

fashion_mnist  = tf.keras.datasets.fashion_mnist
(fashion_mnist_train_x, fashion_mnist_train_y), (fashion_mnist_test_x, fashion_mnist_test_y) = fashion_mnist.load_data() #load data
fashion_mnist_train_x, fashion_mnist_test_x = torch.tensor(fashion_mnist_train_x.reshape(-1, 28*28, order="F")/255, dtype=torch.float), torch.tensor(fashion_mnist_test_x.reshape(-1, 28*28, order="F")/255, dtype=torch.float)
fashion_mnist_train_y, fashion_mnist_test_y = torch.tensor(fashion_mnist_train_y), torch.tensor(fashion_mnist_test_y)


import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML

def get_unsupervised_weights(X, n_hidden, n_epochs, batch_size, prev_weights = None,
        learning_rate=2e-2, precision=1e-30, anti_hebbian_learning_strength=0.4, lebesgue_norm=2.0, rank=2, skip=1):
    sample_sz = X.shape[1]    
    if prev_weights is not None:
        weights = prev_weights.cuda()
        #weights = prev_weights
    else:
        weights = torch.rand((n_hidden, sample_sz), dtype=torch.float).cuda()
        #weights = torch.rand((n_hidden, sample_sz), dtype=torch.float)

    all_weights = torch.zeros((n_epochs*(X.shape[0]//batch_size)//skip + 1, n_hidden, sample_sz), dtype=torch.float)
    a_w_i = 0

    for epoch in range(n_epochs):    
        eps = learning_rate * (1 - epoch / n_epochs)        
        shuffled_epoch_data = X[torch.randperm(X.shape[0]),:]
        for i in range(X.shape[0] // batch_size):
            mini_batch = shuffled_epoch_data[i*batch_size:(i+1)*batch_size,:].cuda()
            #mini_batch = shuffled_epoch_data[i*batch_size:(i+1)*batch_size,:]           
            mini_batch = torch.transpose(mini_batch, 0, 1)            
            sign = torch.sign(weights)            
            W = sign * torch.abs(weights) ** (lebesgue_norm - 1)        
            tot_input=torch.mm(W, mini_batch)  
            
            y = torch.argsort(tot_input, dim=0)
            yl = torch.zeros((n_hidden, batch_size), dtype = torch.float).cuda()            
            #yl = torch.zeros((n_hidden, batch_size), dtype = torch.float)
            yl[y[n_hidden-1,:], torch.arange(batch_size)] = 1.0
            yl[y[n_hidden-rank], torch.arange(batch_size)] =- anti_hebbian_learning_strength            
                    
            xx = torch.sum(yl * tot_input,1)            
            xx = xx.unsqueeze(1)                    
            xx = xx.repeat(1, sample_sz)                            
            ds = torch.mm(yl, torch.transpose(mini_batch, 0, 1)) - xx * weights            
            
            nc = torch.max(torch.abs(ds))            
            if nc < precision:
              nc = precision            
            weights += eps*(ds/nc)
            if a_w_i % skip == 0:
              all_weights[a_w_i//skip] = weights
            a_w_i += 1
        print("Done: " + str(epoch))
    return all_weights


def animate_unsupervised_weights(all_weights, n_hidden, n_epochs, batch_size, n_rows, n_cols, dimensions, dataset_name, starting_from = 0, cmap=None):
    all_weights = all_weights.cpu().numpy().reshape((-1, n_hidden,)+dimensions, order="F")
    fig = plt.figure(figsize=(10, 6))
    plt.axis('off')
    if n_rows*n_cols < n_hidden:
        all_weights = all_weights[:, np.random.choice(all_weights.shape[1], n_rows*n_cols, replace=False)]
    if len(dimensions) == 3:
        HM = np.zeros((n_rows*dimensions[0], n_cols*dimensions[1], dimensions[2]))
    else:
        HM  = np.zeros((n_rows*dimensions[0], n_cols*dimensions[1]))
    
    def animate(i):
        weights = all_weights[i]
        nc=np.amax(np.absolute(weights))
        for idx in range(n_cols * n_rows):
            x, y = idx % n_cols, idx // n_cols
            ma = weights[idx].max()
            mi = weights[idx].min()
            if len(dimensions) > 2:
                HM[y*dimensions[0]:(y+1)*dimensions[0],x*dimensions[1]:(x+1)*dimensions[1]]= (weights[idx] - mi)/(ma - mi)
            else:
                HM[y*dimensions[0]:(y+1)*dimensions[0],x*dimensions[1]:(x+1)*dimensions[1]]= weights[idx]
        plt.clf()
        nc=np.amax(np.absolute(HM))
        
        if cmap is not None:
            im=plt.imshow(HM, cmap=cmap, vmin=-nc, vmax=nc)
            fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
        else:
            im = plt.imshow(HM)

        fig.canvas.draw()
        plt.title("Unsupervised Weights: "+dataset_name + "\n Hidden Units: "+str(n_hidden) + " ("+str(n_rows*n_cols)+" shown)"+"\n Iteration:"+str(starting_from + i))
        print("Done: " + str(i) + " of " + str(len(all_weights)))      
        return fig

    anim = animation.FuncAnimation(fig, animate, frames=all_weights.shape[0])
    anim.save(dataset_name + "_" + str(n_hidden) + "h_" + str(n_epochs)+"e_" + str(batch_size) + "bs" +".mp4")



def train_and_animate(train_data, n_epochs, n_hidden, batch_size, n_rows, n_cols, dimensions, dataset_name, prev_weights = None, starting_from = 0, skip=1, cmap=None):
    all_weights = get_unsupervised_weights(train_data, n_hidden=n_hidden, n_epochs=n_epochs, batch_size=batch_size, prev_weights = prev_weights, skip=skip)
    try:
        if cmap is not None:
            animate_unsupervised_weights(all_weights, n_hidden, n_epochs, batch_size, n_rows, n_cols, dimensions, dataset_name, starting_from, cmap=cmap)
        else:
            animate_unsupervised_weights(all_weights, n_hidden, n_epochs, batch_size, n_rows, n_cols, dimensions, dataset_name, starting_from)
    except:
        return all_weights, -1
    return all_weights, starting_from+len(all_weights)

cifar10_weights, iteration = train_and_animate(cifar10_train_x, 50, 100, 100, 10, 10, (32, 32, 3), "CIFAR10", skip=50)

mnist_weights, iteration = train_and_animate(mnist_train_x, 2, 100, 100, 10, 10, (28, 28), "MNIST", skip=10, cmap="bwr")

cifar10_weights_lg, iteration = train_and_animate(cifar10_train_x, 30, 1600, 100, 10, 10, (32, 32, 3), "CIFAR10", skip=50)

animate_unsupervised_weights(cifar10_weights, 100, 50, 101, 10, 10, (32, 32, 3), "test_adsfasdf2")

mnist_weights_2, iteration = train_and_animate(mnist_train_x, 2, 1600, 100, 10, 10, (28, 28), "MNIST_TOO_MANY_HIDDEN", skip=10, cmap="bwr")

cifar10_weights_2, iteration = train_and_animate(cifar10_train_x, 29, 64, 100, 8, 8, (32, 32, 3), "CIFAR10", skip=50)

cifar10_weights_3, iteration = train_and_animate(cifar10_train_x, 10, 100, 4, 10, 10, (32, 32, 3), "CIFAR10", skip=50 * 25)

cifar10_weights_3, iteration = train_and_animate(cifar10_train_x, 10, 400, 4, 10, 10, (32, 32, 3), "CIFAR10-lots_of-hidden-batch-small-hidden", skip=50 * 5)

final_weights_cifar_10 = cifar10_weights_3[-2]



import keras
import keras.backend as K

#testing good cifar10 bio model vs standard
standard_cifar10_model = keras.Sequential([
    keras.layers.Dense(400, input_shape=(3072,)),                   
    keras.layers.Dense(10, activation="softmax")
])
standard_cifar10_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
history_standard_cifar10 = standard_cifar10_model.fit(cifar10_train_x, cifar10_train_y, batch_size=256, epochs=20, validation_data=(cifar10_test_x, cifar10_test_y))


bio_cifar10_model = keras.Sequential([                   
    keras.layers.Dense(10, input_shape=(400, ), activation="softmax")
])
bio_cifar10_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
history_bio_cifar10 = bio_cifar10_model.fit(augmented_train_cifar_10_400h, cifar10_train_y, batch_size=256, epochs=20, validation_data=(augmented_test_cifar_10_400h, cifar10_test_y))

def compare_models(history_standard, history_bio):
  fig = plt.figure(figsize=(8, 8))
  for metric in ["sparse_categorical_accuracy", "val_sparse_categorical_accuracy"]:
    plt.plot(history_standard.history[metric], label="ANN_"+metric)
    plt.plot(history_bio.history[metric], label="BIO_"+metric)
  plt.legend()
  plt.show()

compare_models(history_standard_cifar10, history_bio_cifar10)

mnist_weights, iteration = train_and_animate(mnist_train_x, 2, 400, 100, 10, 10, (28, 28), "MNIST", skip=10, cmap="bwr")
final_weights_mnist = mnist_weights[-2]

augmented_train_mnist = torch.mm(mnist_train_x, final_weights_mnist.t())
augmented_test_mnist = torch.mm(mnist_test_x, final_weights_mnist.t())

#testing good cifar10 bio model vs standard
standard_mnist_model = keras.Sequential([
    keras.layers.Dense(400, input_shape=(28*28,)),                   
    keras.layers.Dense(10, activation="softmax")
])
standard_mnist_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
history_standard_mnist = standard_mnist_model.fit(mnist_train_x, mnist_train_y, batch_size=256, epochs=20, validation_data=(mnist_test_x, mnist_test_y))

bio_mnist_model = keras.Sequential([                   
    keras.layers.Dense(10, input_shape=(100, ), activation="softmax")
])
bio_mnist_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
history_bio_mnist = bio_mnist_model.fit(augmented_train_mnist, mnist_train_y, batch_size=256, epochs=20, validation_data=(augmented_test_mnist, mnist_test_y))

compare_models(history_standard_mnist, history_bio_mnist)

fashion_mnist_weights, iteration = train_and_animate(fashion_mnist_train_x, 5, 400, 50, 10, 10, (28, 28), "FASHION_MNIST", skip=50, cmap="bwr")

final_weights_fashion_mnist = fashion_mnist_weights[-2]
augmented_train_fashion_mnist_x = torch.mm(fashion_mnist_train_x, final_weights_fashion_mnist.t())
augmented_test_fashion_mnist_x = torch.mm(fashion_mnist_test_x, final_weights_fashion_mnist.t())


#testing good cifar10 bio model vs standard
standard_fashion_mnist_model = keras.Sequential([
    keras.layers.Dense(400, input_shape=(28*28,)),                   
    keras.layers.Dense(10, activation="softmax")
])
standard_fashion_mnist_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
history_standard_fashion_mnist = standard_fashion_mnist_model.fit(fashion_mnist_train_x, fashion_mnist_train_y, batch_size=256, epochs=20, validation_data=(fashion_mnist_test_x, fashion_mnist_test_y))

bio_fashion_mnist_model = keras.Sequential([                   
    keras.layers.Dense(10, input_shape=(400, ), activation="softmax")
])
bio_fashion_mnist_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
history_bio_fashion_mnist = bio_fashion_mnist_model.fit(augmented_train_fashion_mnist_x, fashion_mnist_train_y, batch_size=256, epochs=20, validation_data=(augmented_test_fashion_mnist_x, fashion_mnist_test_y))

compare_models(history_standard_fashion_mnist, history_bio_fashion_mnist)

mnist_weights_small, iteration = train_and_animate(mnist_train_x, 2, 9, 10, 3, 3, (28, 28), "MNIST", skip=100, cmap="bwr")

final_weights_mnist_small = mnist_weights_small[-2]
augmented_train_mnist_small_x = torch.mm(mnist_train_x, final_weights_mnist_small.t())
augmented_test_mnist_small_x = torch.mm(mnist_test_x, final_weights_mnist_small.t())


#testing good cifar10 bio model vs standard
standard_mnist_model_small = keras.Sequential([
    keras.layers.Dense(9, input_shape=(28*28,)),                   
    keras.layers.Dense(10, activation="softmax")
])
standard_mnist_model_small.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
history_standard_mnist_small = standard_mnist_model_small.fit(mnist_train_x, mnist_train_y, batch_size=256, epochs=20, validation_data=(mnist_test_x, mnist_test_y))

bio_mnist_model_small = keras.Sequential([                   
    keras.layers.Dense(10, input_shape=(9, ), activation="softmax")
])
bio_mnist_model_small.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
history_bio_mnist_small = bio_mnist_model_small.fit(augmented_train_mnist_small_x, mnist_train_y, batch_size=256, epochs=20, validation_data=(augmented_test_mnist_small_x, mnist_test_y))

compare_models(history_standard_mnist_small, history_bio_mnist_small)



def get_unsupervised_weights_2(X, n_hidden, n_epochs, batch_size, prev_weights = None,
        learning_rate=2e-2, precision=1e-30, anti_hebbian_learning_strength=0.4, lebesgue_norm=2.0, rank=2):
    sample_sz = X.shape[1]    
    weights = torch.rand((n_hidden, sample_sz), dtype=torch.float).cuda()
    for epoch in range(n_epochs):    
        eps = learning_rate * (1 - epoch / n_epochs)        
        shuffled_epoch_data = X[torch.randperm(X.shape[0]),:]
        for i in range(X.shape[0] // batch_size):
            mini_batch = shuffled_epoch_data[i*batch_size:(i+1)*batch_size,:].cuda()            
            mini_batch = torch.transpose(mini_batch, 0, 1)            
            sign = torch.sign(weights)            
            W = sign * torch.abs(weights) ** (lebesgue_norm - 1)        
            tot_input=torch.mm(W, mini_batch)            
            
            y = torch.argsort(tot_input, dim=0)            
            yl = torch.zeros((n_hidden, batch_size), dtype = torch.float).cuda()
            yl[y[n_hidden-1,:], torch.arange(batch_size)] = 1.0
            yl[y[n_hidden-rank], torch.arange(batch_size)] =- anti_hebbian_learning_strength            
                    
            xx = torch.sum(yl * tot_input,1)            
            xx = xx.unsqueeze(1)                    
            xx = xx.repeat(1, sample_sz)                            
            ds = torch.mm(yl, torch.transpose(mini_batch, 0, 1)) - xx * weights            
            
            nc = torch.max(torch.abs(ds))            
            if nc < precision:
              nc = precision            
            weights += eps*(ds/nc)
        print("Done: " + str(epoch))
    return weights.cpu()

def get_final_mnist_weights(n_hidden):
    return get_unsupervised_weights_2(mnist_train_x, n_hidden, 10, 50)


def train_standard_mnist_model(n_hidden):
    #testing good cifar10 bio model vs standard
    model = keras.Sequential([
        keras.layers.Dense(n_hidden, input_shape=(28*28,)),                   
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    history = model.fit(mnist_train_x, mnist_train_y, batch_size=256, epochs=20, validation_data=(mnist_test_x, mnist_test_y))
    return history.history

def train_bio_mnist_model(n_hidden, final_mnist_weights):
    augmented_mnist_train_x = torch.mm(mnist_train_x, final_mnist_weights.t())
    augmented_mnist_test_x = torch.mm(mnist_test_x, final_mnist_weights.t())

    bio_model = keras.Sequential([                   
        keras.layers.Dense(10, input_shape=(n_hidden, ), activation="softmax")
    ])
    bio_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    history_bio = bio_model.fit(augmented_mnist_train_x, mnist_train_y, batch_size=256, epochs=20, validation_data=(augmented_mnist_test_x, mnist_test_y))
    return history_bio.history

fib = range(7, 13)
standard_val_accuracies = []
bio_val_accuracies = []

for n_hidden in fib:
    final_mnist_weights = get_final_mnist_weights(n_hidden)
    standard_history = train_standard_mnist_model(n_hidden)
    bio_history = train_bio_mnist_model(n_hidden, final_mnist_weights)
    standard_val_accuracies.append(standard_history["val_sparse_categorical_accuracy"])
    bio_val_accuracies.append(bio_history["val_sparse_categorical_accuracy"])

max_standard_val_accuracy = [i[-1] for i in standard_val_accuracies]
max_bio_val_accuracy = [i[-1] for i in bio_val_accuracies]

plt.figure(figsize=(10, 10))
plt.plot(fib, max_standard_val_accuracy, label="ANN_VAL_ACCURACY")
plt.plot(fib, max_bio_val_accuracy, label="BIO_VAL_ACCURACY")
plt.xlabel("# hidden units")
plt.ylabel("Validation Accuracy")
plt.title("Hidden Units vs. Validation Accuracy")
plt.legend()
plt.show()


random_matrix = np.random.normal(0, 1, size=(1000, 784))
random_matrix_mnist_train_x = torch.mm(mnist_train_x, torch.tensor(random_matrix.T, dtype=torch.float))
random_matrix_mnist_test_x = torch.mm(mnist_test_x, torch.tensor(random_matrix.T, dtype=torch.float))
                                      

random_model = keras.Sequential([                   
    keras.layers.Dense(10, input_shape=(1000, ), activation="softmax")
])
random_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
history_random = random_model.fit(random_matrix_mnist_train_x, mnist_train_y, batch_size=256, epochs=20, validation_data=(random_matrix_mnist_test_x, mnist_test_y))

model = keras.Sequential([
      keras.layers.Dense(1000, input_shape=(28*28,)),                   
      keras.layers.Dense(10, activation="softmax")
  ])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
history_standard = model.fit(mnist_train_x, mnist_train_y, batch_size=256, epochs=20, validation_data=(mnist_test_x, mnist_test_y))



fig = plt.figure(figsize=(8, 8))
for metric in ["sparse_categorical_accuracy", "val_sparse_categorical_accuracy"]:
  plt.plot(history_standard.history[metric], label="ANN_"+metric)
  plt.plot(history_random.history[metric], label="RANDOM_"+metric)
plt.legend()
plt.ylabel("Accuracy")
plt.show()

mnist_weights_tiny, iteration = train_and_animate(mnist_train_x, 2, 5, 4, 2, 2, (28, 28), "MNIST-tiny", skip=100, cmap="bwr")