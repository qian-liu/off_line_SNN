import cnn_utils as cnnu
import itertools
import scipy.io as sio
import numpy as np
import sys

tmp_y = sio.loadmat('mnist.mat')['test_y']
tmp_y = np.argmax(tmp_y, axis=0)


def get_result(spikes, time_window):
    spike_count = list()
    predict_max = -1*np.ones(num_test)
    first_spikes = dur_test*np.ones((num_output,num_test))
    fastest = -1*np.ones(num_test)
    latency = dur_test*np.ones(num_test)
    correct_latency = np.zeros(num_test)
    digit_latency = dur_test*np.ones((num_digit,num_test))
    for i in range(num_output):
        index_i = np.where(spikes[:,0] == i)
        spike_train = spikes[index_i, 1][0]
        
        for key, igroup in itertools.groupby(spike_train, lambda x: x // (dur_test+silence)):
            test_id = int(key)
            if test_id>=num_test:
                test_id = num_test-1
            first_spikes[i][test_id] = list(igroup)[0] - test_id*(dur_test+silence)
        
        ind = np.where(np.mod(spike_train ,(dur_test+silence)) <= time_window)[0]
        temp = np.histogram(spike_train[ind], bins=range(0, (dur_test+silence)*num_test+1,dur_test+silence))[0]
        spike_count.append(temp)
        
    spike_count = np.array(spike_count)
    for i in range(num_test):
        if max(spike_count[:,i]) > 0:
            label = np.argmax(spike_count[:,i])//num_cluster
            predict_max[i] = label
            correct_latency[i] = first_spikes[np.argmax(spike_count[:,i])][i]
        fastest[i] = np.argmin(first_spikes[:,i])//num_cluster
        a = np.reshape(first_spikes[:,i], (num_digit, num_cluster))
        digit_latency[:, i] = a.min(axis=1)
        
    latency = np.min(first_spikes, axis=0)
    return predict_max, fastest, latency, correct_latency, digit_latency
    
    
    
def result_analysis(mat_dir):
    predict_max = -1*np.ones(num_mnist)
    fastest_neuron = -1*np.ones(num_mnist)
    respond_time = dur_test*np.ones(num_mnist)
    correct_latency = dur_test*np.ones(num_mnist)
    digit_latency = dur_test*np.ones((num_digit,num_mnist))

    result_timew = []
    
    for time_w in time_ws:
        for test_offset in range(0, num_mnist, num_test):
            spike_f = '%s/spike_%d.npy'%(mat_dir, test_offset)
            spikes = np.load(spike_f)
            predict, fastest, latency, correct_l, digit_l = get_result(spikes, time_w)
            predict_max[test_offset:test_offset+num_test] = predict
            fastest_neuron[test_offset:test_offset+num_test] = fastest
            respond_time[test_offset:test_offset+num_test] = latency
            correct_latency[test_offset:test_offset+num_test] = correct_l
            digit_latency[:,test_offset:test_offset+num_test] = digit_l
    #         print sum(predict == tmp_y[test_offset:test_offset+num_test])
        result_timew.append(sum(predict_max[:num_mnist] ==  tmp_y[:num_mnist]))#tmp_y[test_offset:test_offset+num_test]))
    return result_timew

dur_test = 1000
num_digit = 10
num_test = 100
num_output = 10
silence = 20
num_cluster = 1


result_dir = sys.argv[1]
num_mnist = int(sys.argv[2]) #10000 
time_ws = map(int, sys.argv[3].split(',')) # for example 100,400,1000 ms
print time_ws
result_timew = result_analysis(result_dir) #noisy_softplus relu
for k in range(len(time_ws)):
    result_str = 'Time window: %d ms, accuracy: %.2f%%'%(time_ws[k],result_timew[k]/float(num_mnist)*100.)
    print result_str
