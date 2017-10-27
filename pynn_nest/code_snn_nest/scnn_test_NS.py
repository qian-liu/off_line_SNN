import numpy as np
import pyNN.nest as p
import relu_utils as alg
import spiking_relu as sr
import cnn_utils as cnnu
import spiking_cnn as scnn
import random
import os.path
import sys
#import matplotlib.cm as cm


import scipy.io as sio
tmp_x = sio.loadmat('mnist.mat')['test_x']
tmp_x = np.transpose(tmp_x, (2, 0, 1))
tmp_x = np.reshape(tmp_x, (tmp_x.shape[0], 28*28), order='F' )

tmp_y = sio.loadmat('mnist.mat')['test_y']
tmp_y = np.argmax(tmp_y, axis=0)

# configuation file when trining in Matlab
#config_file = sys.argv[1]
#f = sio.loadmat(config_file)
#e = f['opts'][0]['numepochs'][0][0][0] #num_epochs
#r = f['opts'][0]['randseed'][0][0][0]    #randseed

# activation function configuation
af_file = sys.argv[1]
f = sio.loadmat(af_file)
tau_syn = f['af'][0]['tau_syn'][0][0][0] * 1000. #ms
scale_K = f['af'][0]['K'][0][0][0]

# trained weights
cnn_file = sys.argv[2]               


dur_test = 1000  #ms
silence = 20  #ms
num_test = 100
test_len = 10000
max_rate = 0  #no limit on the over all sumed rates of an 
cell_params_lif = {'cm': 0.25,      #nF
                   'i_offset': 0.1, #nA
                   'tau_m': 20.0,   #ms
                   'tau_refrac': 1.,#ms
                   'tau_syn_E': tau_syn,#ms
                   'tau_syn_I': tau_syn,#ms
                   'v_reset': -65.0,#mV
                   'v_rest': -65.0, #mV
                   'v_thresh': -50.0#mV
                   }
                    
w_cnn, l_cnn = cnnu.readmat(cnn_file)#('cnn_2.mat')# scaled is the 0.023 nsp training.
predict = np.zeros(test_len)
for offset in range(0, test_len, num_test):
    print 'offset: ', offset
    test = tmp_x[offset:(offset+num_test), :]
    test = test * scale_K
    predict[offset:(offset+num_test)],  spikes= scnn.scnn_test(cell_params_lif, l_cnn, w_cnn, num_test, test, max_rate, dur_test, silence)
    print predict[offset:(offset+num_test)] 
    print sum(predict[offset:(offset+num_test)]==tmp_y[offset:(offset+num_test)]) 
    dir = cnn_file[cnn_file.rfind('/')+1:-4]
    spike_f = '../results/%s/spike_%d.npy'%(dir,offset)
    np.save(spike_f, spikes)
np.save('predict_result',predict) 
print np.sum(np.int16(predict==tmp_y))

