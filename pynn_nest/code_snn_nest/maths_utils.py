'''
Functions to be used in Math tasks.
'''
import numpy as np
import math
import random

#times two vetors in to a matrix
def matrix_times(m, n):
    m_matrix = np.transpose(np.tile(m,(len(n), 1)))
    n_matrix = np.tile(n,(len(m), 1))
    return m_matrix*n_matrix
    
    
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
