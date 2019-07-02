from __future__ import  print_function, division
import mxnet as mx
import math

class DictToObj:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_state(m):
    '''
      This code returns model states
    '''
    if m is None:
        return None
    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state

def load_model_states(path):
    '''
     Load previously learned model
    '''
    checkpoint = mx.nd.load(path)
    m_states = checkpoint['model_states']
    m_params = checkpoint['args']

    return m_states, DictToObj(**m_params)

def update_linear_schedule(trainer, epoch, total_num_epochs, initial_lr):
    '''
        Decreases the learning rate linearly
    '''
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    trainer.set_learning_rate(lr)

