## https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/4
## https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
## https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html


import numpy as np
import scipy.misc as misc
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t 


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        #self.hid_size = 32
        self.rnn = nn.LSTM(
            input_size = 16, ## num of features in the input 
            hidden_size = 32,   ## num of features in the hidden state
            num_layers = 10)
        self.relu = nn.ReLU()
        #self.batchnorm = nn.BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
            
    def forward(self,x):
       # x = self.batchnorm(x)
        r_out, (h_n, h_c) = self.rnn(x)
        r_out = self.relu(r_out)
        
        return r_out


## Assumption of input data dimension is: [batch_size, C, H, W, Z, seq_len]
## Assumption the dimension for PyTorch model is: [batch_size, seq_len, C, H, W, Z]

class CombineRNN(nn.Module):
    def __init__(self, devices):
        super(CombineRNN, self).__init__()

        devices = ['cuda:{}'.format(device) for device in devices]
        self.devices = devices

        self.rnn = RNN().to(self.devices[0])
        #self.hid_size = 16
        self.rnn2 = nn.LSTM(
            input_size = 32, ## num of features in the input 
            hidden_size = 16,  ## num of features in the hidden state
            num_layers = 10).to(self.devices[-1])
        self.linear1 = nn.Linear(16,8).to(self.devices[-1])
        self.relu = nn.ReLU().to(self.devices[-1])
        self.linear2 = nn.Linear(8,1).to(self.devices[-1])

    def forward(self, x):

        x = x.to(self.devices[0])
        seq_len, batch_size, feature_size = x.size()  # batch, time

        r_out = self.rnn(x)

        #print('output of rnn1 is {}'.format(r_out.shape))
        
        r_in = r_out.to(self.devices[-1])
        r_out2, (h_n, h_c) = self.rnn2(r_in)

        #print('output of rnn2 is {}'.format(r_out2.shape))

        r_out_l = self.relu(self.linear1(r_out2[-1,:,:]))
        r_output = self.linear2(r_out_l)

        #print('output sizes of linear1 and linear2 are {} and {}'.format(r_out_l.shape, r_output.shape))
        
        return r_output




    

class PipelineCombineRNN(CombineRNN):
    def __init__(self, devices, split_size):
        super(PipelineCombineRNN, self).__init__(devices)
        self.split_size = split_size

        devices = ['cuda:{}'.format(device) for device in devices]
        self.devices = devices
        
    def forward(self, x):
    
        splits = iter(x.split(self.split_size, dim=1))

        s_next = next(splits)    

        s_next = s_next.to(self.devices[0])
        s_prev = self.rnn(s_next).to(self.devices[-1])
        ret = []

        for s_next in splits:
            s_prev, _ = self.rnn2(s_prev)
            s_prev = self.linear2(self.relu(self.linear1(s_prev[-1,:,:])))
            ret.append(s_prev)

            s_next = s_next.to(self.devices[0])
            s_prev = self.rnn(s_next).to(self.devices[-1])

        s_prev, _ = self.rnn2(s_prev)
        s_prev = self.linear2(self.relu(self.linear1(s_prev[-1,:,:])))
        ret.append(s_prev)

        return torch.cat(ret)
    













