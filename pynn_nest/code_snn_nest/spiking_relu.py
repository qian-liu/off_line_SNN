import numpy as np
# import matplotlib.pyplot as plt
import pyNN.nest as p
import scipy.io as sio
import mnist_utils as mu
import random
import relu_utils as alg
import copy

def plot_spikes(spikes, title):
    fig, ax = plt.subplots()
    ax.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
    plt.show()

def transf(k, x0, y0, curr):
    rate = k*(curr-x0) + y0
    rate[rate<0] = 0
    return rate
    
def rev_transf(k, x0, y0, rate):
    curr = (rate - y0) / k + x0
    return curr

'''
sim = sys.argv[1]
if sim == 'nest':
    import pyNN.nest as p
elif sim == 'spin':
    import spynnaker.pyNN as p
else:
    sys.exit()
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 1.,   # 2.0
                   'tau_syn_E': 1.0,
                   'tau_syn_I': 1.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }
'''

def estimate_kb(cell_params_lif):
    cell_para = copy.deepcopy(cell_params_lif)
    random.seed(0)
    p.setup(timestep=1.0, min_delay=1.0, max_delay=16.0)
    run_s = 10.
    runtime = 1000. * run_s
    max_rate = 1000.
    ee_connector = p.OneToOneConnector(weights=1.0, delays=2.0)    


    pop_list = []
    pop_output = []
    pop_source = []
    x = np.arange(0., 1.01, 0.1)
    count = 0
    trail = 10

    for i in x:
        for j in range(trail): #trails for average
            pop_output.append(p.Population(1, p.IF_curr_exp, cell_para))
            poisson_spikes = mu.poisson_generator(i*max_rate, 0, runtime)
            pop_source.append( p.Population(1, p.SpikeSourceArray, {'spike_times' : poisson_spikes}) )
            p.Projection(pop_source[count], pop_output[count], ee_connector, target='excitatory')
            pop_output[count].record()
            count += 1


    count = 0
    for i in x:
        cell_para['i_offset'] = i
        pop_list.append(p.Population(1, p.IF_curr_exp, cell_para))
        pop_list[count].record()
        count += 1
    pop_list[count-1].record_v()

    p.run(runtime)

    rate_I = np.zeros(count)
    rate_P = np.zeros(count)
    rate_P_max = np.zeros(count)
    rate_P_min = np.ones(count) * 1000.
    for i in range(count):
        spikes = pop_list[i].getSpikes(compatible_output=True)
        rate_I[i] = len(spikes)/run_s
        for j in range(trail):
            spikes = pop_output[i*trail+j].getSpikes(compatible_output=True)
            spike_num = len(spikes)/run_s
            rate_P[i] += spike_num
            if spike_num > rate_P_max[i]:
                rate_P_max[i] = spike_num
            if spike_num < rate_P_min[i]:
                rate_P_min[i] = spike_num
        rate_P[i] /= trail
    '''
    #plot_spikes(spikes, 'Current = 10. mA')
    plt.plot(x, rate_I, label='current',)
    plt.plot(x, rate_P, label='Poisson input')
    plt.fill_between(x, rate_P_min, rate_P_max, facecolor = 'green', alpha=0.3)
    '''
    x0 = np.where(rate_P>1.)[0][0]
    x1 = 4
    k = (rate_P[x1] - rate_P[x0])/(x[x1]-x[x0])
    '''
    plt.plot(x, k*(x-x[x0])+rate_P[x0], label='linear')
    plt.legend(loc='upper left', shadow=True)
    plt.grid('on')
    plt.show()
    '''
    p.end()
    return k, x[x0], rate_P[x0]

def w_adjust(dbnet, cell_para, SUM_rate=2000., lim_rate = 20.):
    train_x = np.copy(dbnet['train_x'])
    k, x0, y0 = estimate_kb(cell_para)
    for i in range(train_x.shape[0]):
        train_x[i] = train_x[i] / sum(train_x[i]) * SUM_rate
    w_list = []
    for i in range(len(dbnet['layer'])):
        w = np.copy(dbnet['layer'][i]['w_up'])
        scale, train_x = scale_to_spike(train_x, w, k, x0, y0, lim_rate)
        w_list.append(w * scale)

    h_num = dbnet['top']['v_num'] - dbnet['top']['label_n']
    w = np.copy(dbnet['top']['w'][:h_num,:])
    scale, train_x = scale_to_spike(train_x, w, k, x0, y0, lim_rate)
    w_list.append(w * scale)
    w = np.copy(np.transpose(dbnet['top']['w'][h_num:,:]))
    scale, train_x = scale_to_spike(train_x, w, k, x0, y0, lim_rate)
    w_list.append(w * scale)
    
    return w_list, k, x0, y0

def scale_to_spike(train_x, w, k, x0, y0, lim_rate):
    curr = alg.ReLU(train_x, w)
    mean = np.mean(curr)
    #std = np.std(curr)
    #x_lim = rev_transf(k, x0, y0, lim_rate) 
    #scale = x_lim * 1000. / (mean + 3 * std)
    x_mid = rev_transf(k, x0, y0, lim_rate) 
    scale = x_mid * 1000. / mean
    curr *= scale/1000.
    out_rate = transf(k, x0, y0, curr)
    '''
    count, edges = np.histogram(curr)
    width = edges[1] - edges[0]
    plt.bar(edges[:-1], count, width=width)
    plt.show()
    '''
    return scale, out_rate
    
def gen_spike_source(data, input_size = 60, SUM_rate=2000., dur_test=1000, silence=200):
    input_size = 28
    spike_source_data = mu.mnist_poisson_gen(data, input_size, input_size, SUM_rate, dur_test, silence)
    return spike_source_data
    
def run_test(w_list, cell_para, spike_source_data):
    pop_list = []
    p.setup(timestep=1.0, min_delay=1.0, max_delay=3.0)
    #input poisson layer
    input_size = w_list[0].shape[0]
    pop_in = p.Population(input_size, p.SpikeSourceArray, {'spike_times' : []})
    for j in range(input_size):
        pop_in[j].spike_times = spike_source_data[j]
    pop_list.append(pop_in)
    
    for w in w_list:        
        pos_w = np.copy(w)
        pos_w[pos_w < 0] = 0
        neg_w = np.copy(w)
        neg_w[neg_w > 0] = 0
        
        output_size = w.shape[1]
        pop_out = p.Population(output_size, p.IF_curr_exp, cell_para)
        p.Projection(pop_in, pop_out, p.AllToAllConnector(weights = pos_w), target='excitatory')
        p.Projection(pop_in, pop_out, p.AllToAllConnector(weights = neg_w), target='inhibitory')
        pop_list.append(pop_out)
        pop_in = pop_out

    pop_out.record()
    run_time = np.ceil(np.max(spike_source_data)[0]/1000.)*1000
    p.run(run_time)
    spikes = pop_out.getSpikes(compatible_output=True)
    return spikes

def counter(data, left_edge, dur):
    count = []
    for l in left_edge:
        temp = np.where((data >= l) & (data < l+dur))
        count.append(temp[0].shape[0])
    return count
    
def test_sdbn():

    return result





