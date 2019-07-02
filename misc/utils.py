import os
import numpy as np
import gym
import glob
import json
from collections import deque, OrderedDict
import psutil
import re
import csv

def identity(x):
    '''
        identity function 
    '''
    return x

def get_action_info(action_space, obs_space = None):
    '''
        This fucntion returns info about type of actions.
    '''
    space_type = action_space.__class__.__name__

    if action_space.__class__.__name__ == "Discrete":
            num_actions = action_space.n

    elif action_space.__class__.__name__ == "Box":
            num_actions = action_space.shape[0]

    elif action_space.__class__.__name__ == "MultiBinary":
            num_actions = action_space.shape[0]
    
    else:
        raise NotImplementedError
    
    return num_actions, space_type

def create_dir(log_dir, ext = '*.monitor.csv', cleanup = False):

    '''
        Setup checkpoints dir
    '''

    try:
        os.makedirs(log_dir)

    except OSError:
        if cleanup == True:
            files = glob.glob(os.path.join(log_dir, '*.'))

            for f in files:
                os.remove(f)

def dump_to_json(path, data):
    '''
      Write json file
    '''
    with open(path, 'w') as f:
        json.dump(data, f)

def explained_variance(ypred, y):
    """
        Computes fraction of variance that ypred explains about y.
        Returns 1 - Var[y-ypred] / Var[y]

        interpretation:
            ev=0  =>  might as well have predicted zero
            ev=1  =>  perfect prediction
            ev<0  =>  worse than just predicting zero
        code : https://github.com/openai/baselines/blob/master/baselines/common/math_util.py
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

class EpisodeStats:
    def __init__(self, nsteps, nenvs):
        self.episode_rewards = []
        for i in range(nenvs):
            self.episode_rewards.append([])
        self.lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
        self.rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
        self.nsteps = nsteps
        self.nenvs = nenvs

    def feed(self, rewards, masks):
        rewards = np.reshape(rewards, [self.nenvs, self.nsteps])
        masks = np.reshape(masks, [self.nenvs, self.nsteps])
        for i in range(0, self.nenvs):
            for j in range(0, self.nsteps):
                self.episode_rewards[i].append(rewards[i][j])
                if masks[i][j]:
                    l = len(self.episode_rewards[i])
                    s = sum(self.episode_rewards[i])
                    self.lenbuffer.append(l)
                    self.rewbuffer.append(s)
                    self.episode_rewards[i] = []

    def mean_length(self):
        if self.lenbuffer:
            return np.mean(self.lenbuffer)
        else:
            return 0  # on the first params dump, no episodes are finished

    def mean_reward(self):
        if self.rewbuffer:
            return np.mean(self.rewbuffer)
        else:
            return 0

def csv_writer(fname, fieldnames):
    '''
        Write a csv file
    '''
    csv_file = open(fname, mode='w')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    return writer

def safemean(xs):
    '''
        Avoid division error when calculate the mean (in our case if
        epinfo is empty returns np.nan, not return an error)
    '''
    return np.nan if len(xs) == 0 else np.mean(xs)