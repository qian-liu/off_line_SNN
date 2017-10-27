#import matplotlib.pyplot as plt
import numpy as np
import pyNN.nest as p
import relu_utils as alg
import spiking_relu as sr
import random
import mnist_utils as mu
import os.path
import sys
import cnn_utils as cnnu

def conv_conn(in_size, out_size, w):
    conn_list_exci = []
    conn_list_inhi = []
    #conn_list = [] #nest works with mixed exci and inhi connections
    k_size = in_size - out_size + 1
    for x_ind in range(out_size):
        for y_ind in range(out_size):
            out_ind = x_ind * out_size + y_ind
            for kx in range(k_size):
                for ky in range(k_size):
                    in_ind = (x_ind+kx) * in_size + (y_ind+ky)
                    weight = w[k_size-1-ky][k_size-1-kx] #transpose(w)
                    if weight>0:
                        conn_list_exci.append((in_ind, out_ind, weight, 1.)) 
                    elif weight<0:
                        conn_list_inhi.append((in_ind, out_ind, weight, 1.)) 
                    #conn_list.append((in_ind, out_ind, weight, 1.))
    return conn_list_exci, conn_list_inhi#, conn_list

def pool_conn(in_size, out_size, w):
    conn_list = []
    step = in_size/out_size
    for x_ind in range(out_size):
        for y_ind in range(out_size):
            out_ind = x_ind * out_size + y_ind
            for kx in range(step):
                for ky in range(step):
                    in_ind = (x_ind*step+kx) * in_size + (y_ind*step+ky)
                    conn_list.append((in_ind, out_ind, w, 1.))
    return conn_list

def out_conn(w):
    conn_list_exci = []
    conn_list_inhi = []
    #conn_list = [] #nest works with mixed exci and inhi connections
    for j in range(w.shape[0]):
        for i in range(w.shape[1]):
            weight = w[j][i]
            if weight>0:
                conn_list_exci.append((i, j, weight, 1.)) 
            elif weight<0:
                conn_list_inhi.append((i, j, weight, 1.)) 
            #conn_list.append((i, j, weight, 1.))
    return conn_list_exci, conn_list_inhi#, conn_list
    
    for x_ind in range(out_size):
        for y_ind in range(out_size):
            out_ind = x_ind * out_size + y_ind
            for kx in range(k_size):
                for ky in range(k_size):
                    in_ind = (x_ind+kx) * in_size + (y_ind+ky)
                    weight = w[k_size-1-ky][k_size-1-kx] #transpose(w)
                    if weight>0:
                        conn_list_exci.append((in_ind, out_ind, weight, 1.)) 
                    elif weight<0:
                        conn_list_inhi.append((in_ind, out_ind, weight, 1.)) 
                    #conn_list.append((in_ind, out_ind, weight, 1.))
    return conn_list_exci, conn_list_inhi#, conn_list

def conv_pops(pop1, pop2, w):
    in_size = int(np.sqrt(pop1.size))
    out_size = int(np.sqrt(pop2.size))
    conn_exci, conn_inhi = conv_conn(in_size, out_size, w)
    if len(conn_exci)>0:
        p.Projection(pop1, pop2, p.FromListConnector(conn_exci), target='excitatory')
    if len(conn_inhi)>0:
        p.Projection(pop1, pop2, p.FromListConnector(conn_inhi), target='inhibitory')
    return

def pool_pops(pop1, pop2, w):
    in_size = int(np.sqrt(pop1.size))
    out_size = int(np.sqrt(pop2.size))
    conn_exci = pool_conn(in_size, out_size, w)
    if len(conn_exci)>0:
        p.Projection(pop1, pop2, p.FromListConnector(conn_exci), target='excitatory')
    return

def out_pops(pop_list, pop2, w_layer):
    in_size = pop_list[0].size
    out_size = pop2.size
    for i in range(len(pop_list)):
        w = w_layer[:,i*in_size:(i+1)*in_size]
        conn_exci, conn_inhi = out_conn(w)
        if len(conn_exci)>0:
            p.Projection(pop_list[i], pop2, p.FromListConnector(conn_exci), target='excitatory')
        if len(conn_inhi)>0:
            p.Projection(pop_list[i], pop2, p.FromListConnector(conn_inhi), target='inhibitory')
    return

def init_inputlayer(input_size, data, sum_rate, dur, silence):
    pop_list = []
    pop = p.Population(input_size*input_size, p.SpikeSourceArray, {'spike_times' : []})
    spike_source_data = sr.gen_spike_source(data,SUM_rate=sum_rate, dur_test = dur, silence=silence)
    for j in range(input_size*input_size):
        pop[j].spike_times = spike_source_data[j]
    pop_list.append(pop)
    return pop_list
    
def construct_layer(cell_params_lif, pop_list_in, mode, k_size, w_layer):
    max_F = 140. #150.
    max_curr = 1. #1.
    syn = 5.*0.96
    in_num = len(pop_list_in) #populations number in previous layer
    in_size = int(np.sqrt(pop_list_in[0].size)) #in_size*in_size = neuron_num per pop in the previous layer
    pop_layer = []
    if mode > 0: #convoluational layer
        out_num = mode #populations number in current layer
        #print in_num, out_num
        out_size = in_size - k_size + 1
        for j in range(out_num):
            pop_layer.append(p.Population(out_size*out_size, p.IF_curr_exp, cell_params_lif))
            for i in range(in_num):
                conv_pops(pop_list_in[i], pop_layer[j], w_layer[i][j])# * 1000./max_F * max_curr / syn)
    elif mode == 0: #pooling layer
        out_num = in_num #populations number in current layer
        #print in_num, out_num
        out_size = in_size/k_size
        for j in range(out_num):
            pop_layer.append(p.Population(out_size*out_size, p.IF_curr_exp, cell_params_lif))
            pool_pops(pop_list_in[j], pop_layer[j], w_layer[0][0])# * 1000./max_F * max_curr / syn)
    elif mode == -1: #top layer
        out_size = k_size
        #print out_size
        pop_layer.append(p.Population(out_size, p.IF_curr_exp, cell_params_lif))
        out_pops(pop_list_in, pop_layer[0], w_layer)# * 1000./max_F * max_curr / syn)
    return pop_layer
    
def scnn_test(cell_params_lif, l_cnn, w_cnn, num_test, test, max_rate, dur_test, silence):
    p.setup(timestep=1.0, min_delay=1.0, max_delay=3.0)
    L = l_cnn
    random.seed(0)
    input_size = L[0][1]
    pops_list = []
    pops_list.append(init_inputlayer(input_size, test[:num_test, :], max_rate, dur_test, silence))
    print 'SCNN constructing...'
    for l in range(len(w_cnn)):
        pops_list.append(construct_layer(cell_params_lif, pops_list[l], L[l+1][0], L[l+1][1], w_cnn[l]))
    result = pops_list[-1][0]
    result.record()
    
    print 'SCNN running...'
    p.run((dur_test+silence)*num_test)
    spike_result = result.getSpikes(compatible_output=True)
    p.end()
    
    print 'analysing...'
    spike_result_count = count_spikes(spike_result, 10, num_test, dur_test, silence)
    predict = np.argmax(spike_result_count, axis=0)
    
#     prob = np.exp(spike_result_count)/np.sum(np.exp(spike_result_count), axis=0)
    return predict, spike_result
    
def count_spikes(spikes, num_neuron, num_test, dur_test, silence):
    spike_count = []
    for i in range(num_neuron):
        index_i = np.where(spikes[:,0] == i)
        spike_train = spikes[index_i, 1]
        temp = sr.counter(spike_train, range(0, (dur_test+silence)*num_test,dur_test+silence), dur_test)
        spike_count.append(temp)
    spike_count = np.array(spike_count)/(dur_test / 1000.)
    spike_count = np.array(spike_count)
    return spike_count
