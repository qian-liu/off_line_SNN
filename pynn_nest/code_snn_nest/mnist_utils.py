'''
Functions to be used in MNIST related tasks.
'''

import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import math
import random

# get training data
def get_train_data():
    dir_path = './'
    file_name = dir_path + 'train-images.idx3-ubyte'
    print file_name
    
    f = open(file_name, "rb")
    magic_number, list_size, image_hight, image_width  = np.fromfile(f, dtype='>i4', count=4)
    train_x = np.fromfile(f, dtype='>u1', count=list_size*image_hight*image_width)
    train_x = np.reshape(train_x, (list_size,image_hight*image_width))
    f.close()
    
    file_name = dir_path + 'train-labels.idx1-ubyte'
    f = open(file_name, "rb")
    magic_number, list_size = np.fromfile(f, dtype='>i4', count=2)
    train_y = np.fromfile(f, dtype='>u1', count=list_size*image_hight*image_width)
    f.close()
    
    return np.double(train_x), np.double(train_y)

# get testing data
def get_test_data():
    dir_path = './'
    file_name = dir_path + 't10k-images.idx3-ubyte'
    f = open(file_name, "rb")
    magic_number, list_size, image_hight, image_width  = np.fromfile(f, dtype='>i4', count=4)
    test_x = np.fromfile(f, dtype='>u1', count=list_size*image_hight*image_width)
    test_x = np.reshape(test_x, (list_size,image_hight*image_width))
    f.close()
    
    file_name = dir_path +  't10k-labels.idx1-ubyte'
    f = open(file_name, "rb")
    magic_number, list_size = np.fromfile(f, dtype='>i4', count=2)
    test_y = np.fromfile(f, dtype='>u1', count=list_size*image_hight*image_width)
    f.close()
    
    return np.double(test_x), np.double(test_y)
    
#plot a MNIST digit
def plot_digit(img_raw):
    #img_raw = np.uint8(img_raw)
    plt.figure(figsize=(5,5))
    im = plt.imshow(np.reshape(img_raw,(28,28)), cmap=cm.gray_r,interpolation='none')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    
def nextTime(rateParameter):
    return -math.log(1.0 - random.random()) / rateParameter
    #random.expovariate(rateParameter)
def poisson_generator(rate, t_start, t_stop):
    poisson_train = []
    if rate > 0:
        next_isi = nextTime(rate)*1000.
        last_time = next_isi + t_start
        while last_time  < t_stop:
            poisson_train.append(last_time)
            next_isi = nextTime(rate)*1000.
            last_time += next_isi
    return poisson_train


# In[23]:

def mnist_poisson_gen(image_list, image_height, image_width, max_freq, duration, silence):
    if max_freq > 0:
        for i in range(image_list.shape[0]):
            image_list[i] = image_list[i]/sum(image_list[i])*max_freq
    
    spike_source_data = [[] for i in range(image_height*image_width)]
    
    for i in range(image_list.shape[0]):
        t_start = i*(duration+silence)
        t_stop = t_start+duration
        for j in range(image_height*image_width):
            spikes = poisson_generator(image_list[i][j], t_start, t_stop)
            if spikes != []:
                spike_source_data[j].extend(spikes)
            
    return spike_source_data
