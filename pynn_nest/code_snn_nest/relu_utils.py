'''
Functions to be used in ReLU tasks.
'''

import mnist_utils as mu
import maths_utils as matu
import numpy as np
import pickle

def save_dict(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
        
def init_para(vis_num, hid_num, eta):
    para = {}
    para['h_num'] = hid_num
    para['v_num'] = vis_num
    para['eta'] = eta
    w = np.random.normal(0,0.01, vis_num*hid_num)
    para['w'] = w.reshape((vis_num,hid_num))
    return para

def plot_recon(digit_img, para):
    data_v = np.array(digit_img).astype(float)
    data_h, gibbs_v, gibbs_h = sampling(para, data_v)
    mu.plot_digit(gibbs_v)
    
def update_batch_cd1(para, data_v):
    eta = para['eta']
    max_bsize = data_v.shape[0]
    data_h, gibbs_v, gibbs_h = sampling(para, data_v)
    
    pos_delta_w = np.zeros((para['v_num'], para['h_num']))
    neg_delta_w = np.zeros((para['v_num'], para['h_num']))
    for i in range(max_bsize):
        pos_delta_w += matu.matrix_times(data_v[i], data_h[i])
        neg_delta_w += matu.matrix_times(gibbs_v[i], gibbs_h[i])    
    delta_w_pos = eta * pos_delta_w/np.float(max_bsize)
    delta_w_neg = eta * neg_delta_w/np.float(max_bsize)
    para['w'] += delta_w_pos
    para['w'] -= delta_w_neg
    #print delta_w_pos.max(), delta_w_neg.max()
    return para
    
def ReLU(data, weight):
    sum_data = np.dot(data, weight)
    sum_data[sum_data < 0] = 0
    return sum_data

def sampling(para, data_v):
    w = para['w']
    h0 = ReLU(data_v, w)
    v1 = ReLU(h0, w.transpose())
    h1 = ReLU(v1, w)
    return h0, v1, h1
    
def init_label_dbn(train_data, label_data, nodes, eta=1e-3, batch_size=10, epoc=5):
    if train_data.shape[1] != nodes[0]:
        print 'Dimention of train_data has to equal to the input layer size.'
        exit()
    elif label_data.shape[1] != nodes[-1]:
        print 'Dimention of label_data has to equal to the output layer size.'
        exit()
    elif train_data.shape[0] != label_data.shape[0]:
        print 'The amount of data and label should be the same.'
        exit()
    dbnet = {}
    dbnet['train_x'] = train_data
    dbnet['train_y'] = label_data
    dbnet['nodes'] = nodes
    dbnet['batch_size'] = batch_size
    dbnet['epoc'] = epoc
    
    para_list = []
    for i in range(len(nodes) - 3):   #bottom up
        para_list.append(init_para(nodes[i], nodes[i+1], eta))
    para_top = init_para(nodes[-3] + nodes[-1], nodes[-2], eta)
    para_top['label_n'] = 10
    dbnet['layer'] = para_list
    dbnet['top'] = para_top
    
    return dbnet

def RBM_train(para, epoc, batch_size, train_data):
    train_num = train_data.shape[0]
    for iteration in range(epoc):
        for k in range(0,train_num,batch_size):
            max_bsize = min(train_num-k, batch_size)
            data_v = train_data[k:k+max_bsize]
            para = update_batch_cd1(para, data_v)
    return para

def greedy_train(dbnet):
    batch_size = dbnet['batch_size']
    train_size = dbnet['train_x'].shape[0]
    drop_out = 0.5
    train_index = np.random.choice(train_size, train_size*drop_out, replace=False)
    train_data = dbnet['train_x'][train_index]
    train_label = dbnet['train_y'][train_index]
    for i in range(len(dbnet['layer'])):   #bottom up
        dbnet['layer'][i] = RBM_train(dbnet['layer'][i], dbnet['epoc'], batch_size, train_data)
        train_data = ReLU(train_data, dbnet['layer'][i]['w'])
    train_data = np.append(train_data, train_label, axis=1)
    dbnet['top'] = RBM_train(dbnet['top'], dbnet['epoc'], batch_size, train_data)
    return dbnet
    

def update_unbound_w(w_up, w_down, d_vis):
    bsize = d_vis.shape[0]
    delta_w = 0
    d_hid = ReLU(d_vis, w_up)
    g_vis = ReLU(d_hid, w_down)
    for ib in range(bsize):
        delta_w += matu.matrix_times(d_hid[ib], d_vis[ib]-g_vis[ib])
    delta_w /= np.float(bsize)
    return delta_w, d_hid
    
def fine_train(dbnet):
    batch_size = dbnet['batch_size']
    train_data = dbnet['train_x']
    train_num = train_data.shape[0]
    for i in range(len(dbnet['layer'])):   #bottom up
        dbnet['layer'][i]['w_up'] = dbnet['layer'][i]['w']
        dbnet['layer'][i]['w_down'] = np.transpose(dbnet['layer'][i]['w'])
    for iteration in range(dbnet['epoc']):
        for k in range(0,train_num,batch_size):
            max_bsize = min(train_num-k, batch_size)
            d_vis = train_data[k:k+max_bsize]
            label = dbnet['train_y'][k:k+max_bsize]
            #up
            for i in range(len(dbnet['layer'])):   #bottom up
                delta_w, d_vis = update_unbound_w(dbnet['layer'][i]['w_up'], dbnet['layer'][i]['w_down'], d_vis)
                dbnet['layer'][i]['w_down'] += dbnet['layer'][i]['eta'] * delta_w
            #top
            d_vis = np.append(d_vis, label, axis=1)
            dbnet['top'] = update_batch_cd1(dbnet['top'], d_vis)
            d_hid, g_vis, g_hid = sampling(dbnet['top'], d_vis)
            d_vis = g_vis[:, :dbnet['top']['v_num'] - dbnet['top']['label_n']]
            #down
            for i in range(len(dbnet['layer'])-1, -1, -1):   #up down
                delta_w, d_vis = update_unbound_w(dbnet['layer'][i]['w_down'], dbnet['layer'][i]['w_up'], d_vis)
                dbnet['layer'][i]['w_up'] += dbnet['layer'][i]['eta'] * delta_w
    return dbnet

def dbn_recon(dbnet, test):
    temp = test
    top_inputsize = dbnet['top']['v_num'] - dbnet['top']['label_n']
    for i in range(len(dbnet['layer'])):   #bottom up
        temp = ReLU(temp, dbnet['layer'][i]['w_up'])
    top = ReLU(temp, dbnet['top']['w'][:top_inputsize, :])
    label = ReLU(top, np.transpose(dbnet['top']['w'][top_inputsize:, :]))
    temp = np.append(temp, label, axis=1)
    temp = ReLU(temp, dbnet['top']['w'])
    temp = ReLU(temp, np.transpose(dbnet['top']['w']))
    temp = temp[:top_inputsize]
    for i in range(len(dbnet['layer'])-1, -1, -1):   #up down
        temp = ReLU(temp, dbnet['layer'][i]['w_down'])
    recon = temp
    mu.plot_digit(recon)
    predict = np.argmax(label)
    return predict, recon

def greedy_recon(dbnet, test):
    temp = test
    top_inputsize = dbnet['top']['v_num'] - dbnet['top']['label_n']
    for i in range(len(dbnet['layer'])):   #bottom up
        temp = ReLU(temp, dbnet['layer'][i]['w'])
    top = ReLU(temp, dbnet['top']['w'][:top_inputsize, :])
    label = ReLU(top, np.transpose(dbnet['top']['w'][top_inputsize:, :]))
    temp = np.append(temp, label, axis=1)
    temp = ReLU(temp, dbnet['top']['w'])
    temp = ReLU(temp, np.transpose(dbnet['top']['w']))
    temp = temp[:top_inputsize]
    for i in range(len(dbnet['layer'])-1, -1, -1):   #up down
        temp = ReLU(temp, np.transpose(dbnet['layer'][i]['w']))
    recon = temp
    mu.plot_digit(recon)
    predict = np.argmax(label)
    return predict, recon
    
def test_label_data(dbnet, test_data, test_label):
    dbnet['test_x'] = test_data
    dbnet['test_y'] = test_label
    return dbnet

def dbn_test(dbnet):
    temp = dbnet['test_x']
    top_inputsize = dbnet['top']['v_num'] - dbnet['top']['label_n']
    for i in range(len(dbnet['layer'])):   #bottom up
        temp = ReLU(temp, dbnet['layer'][i]['w_up'])
    top = ReLU(temp, dbnet['top']['w'][:top_inputsize, :])
    label = ReLU(top, np.transpose(dbnet['top']['w'][top_inputsize:, :]))
    predict = np.argmax(label, axis=1)
    index = np.where(label.max(axis=1)==0)[0]
    predict[index] = -1
    result = predict == dbnet['test_y']
    result = result.astype(int)
    result[index] = -1
    return predict, result

def dbn_greedy_test(dbnet):
    temp = dbnet['test_x']
    top_inputsize = dbnet['top']['v_num'] - dbnet['top']['label_n']
    for i in range(len(dbnet['layer'])):   #bottom up
        temp = ReLU(temp, dbnet['layer'][i]['w'])
    top = ReLU(temp, dbnet['top']['w'][:top_inputsize, :])
    label = ReLU(top, np.transpose(dbnet['top']['w'][top_inputsize:, :]))
    predict = np.argmax(label, axis=1)
    index = np.where(label.max(axis=1)==0)[0]
    predict[index] = -1
    result = predict == dbnet['test_y']
    result = result.astype(int)
    result[index] = -1
    return predict, result
