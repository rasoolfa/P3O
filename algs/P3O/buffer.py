import numpy as np
import mxnet as mx

class Buffer:
    '''
        FIFO ReplyBuffer
    '''
    def __init__(self, num_envs, nsteps, sample_multiplier = 1,
                 frames_waits=10000, init_size=200, device = mx.cpu()):
        '''
            frames: Sampling from the replay buffer does not start until
            replay buffer has at least that many samples

            init_size: Each loc contains nenv * nsteps frames,
            so replay_size is init_size // (self.nsteps)

            sample_size: defines how many samples per batch which is set to nenv in _init_

        '''
        if sample_multiplier  <= 0:
            sample_multiplier = 1

        # number of env and steps
        self.nenv = num_envs
        self.nsteps = nsteps

        self.sample_multiplier = sample_multiplier
        self.batch_size = sample_multiplier * self.nenv
        self.device = device

        # Sampling from the replay buffer does not start until replay buffer has at least that many samples
        self.frames_waits = frames_waits

        # buffer specs
        self.buff_size = init_size // (self.nsteps)
        self.next_idx = 0
        self.num_in_buffer = 0

        # memory
        self.obs_var = None
        self.actions = None
        self.b_probs  = None
        self.b_logprobs = None
        self.rewards = None
        self.masks = None

    def has_enough_frames(self):
        '''
            This function checks that Sampling from the replay buffer does not
            start until replay buffer has at least that many samples.

            Frames per env, so total (nenv * frames) Frames needed
            Each buffer loc has nenv * nsteps frames
        '''
        return self.num_in_buffer >= (self.frames_waits // self.nsteps)

    def can_sample(self):
        '''
            Check if it can sample from replay buffer
        '''
        return self.num_in_buffer > 0 and self.num_in_buffer > self.batch_size

    def put(self, obs_var, actions, b_probs, b_logprobs, rewards, masks):
        '''
            Add samples to replay buffer
            obs_var [nsteps + 1, nenv, ...]
            actions, rewards, masks [nsteps, nenv]
            b_probs, b_logprobs: [nsteps, nenv, num_actions]
            b_probs and b_logprobs refer to behavioral policy
        '''

        if self.obs_var is None:
            self.obs_var = np.empty([self.buff_size] + list(obs_var.shape), dtype=obs_var.dtype)
            self.actions = mx.nd.empty([self.buff_size] + list(actions.shape)).astype(actions.dtype)
            self.b_probs = mx.nd.empty([self.buff_size] + list(b_probs.shape)).astype(b_probs.dtype)
            self.b_logprobs = mx.nd.empty([self.buff_size] + list(b_logprobs.shape)).astype(b_logprobs.dtype)
            self.rewards = mx.nd.empty([self.buff_size] + list(rewards.shape)).astype(rewards.dtype)
            self.masks = mx.nd.empty([self.buff_size] + list(masks.shape)).astype(masks.dtype)

        self.obs_var[self.next_idx] = obs_var
        self.actions[self.next_idx] = actions.detach()
        self.b_probs[self.next_idx] = b_probs.detach() # b_probs is part of computational graph
        self.b_logprobs[self.next_idx] = b_logprobs.detach() # b_logprobs is part of computational graph
        self.rewards[self.next_idx] = rewards.detach()
        self.masks[self.next_idx] = masks.detach()

        # update idx
        self.next_idx = (self.next_idx + 1) % self.buff_size
        self.num_in_buffer = min(self.buff_size, self.num_in_buffer + 1)


    def take(self, x, idx, envx):
        '''
            This function putogther the samples
        '''
        # X == > (buff_size, steps, env, ...)
        # sample_size equals to nenv
        out = mx.nd.empty([x.shape[1]] + [self.batch_size] + list(x.shape[3:])).astype(x.dtype)

        for i in range(self.batch_size):
            # X   ==> (buff_size, steps, env,...)
            # out ==> (steps, env/sample_size,...)
            out[:,i] = x[idx[i],:, envx[i]]

        return out

    def take_numpy(self, idx, envx):
        '''
            This function putogther the samples
        '''
        # X == > (buff_size, steps, env, ...)
        # sample_size equals to nenv
        out = np.empty([self.obs_var.shape[1]] + [self.batch_size] + list(self.obs_var.shape[3:])).astype(self.obs_var.dtype)

        for i in range(self.batch_size):
            # self.obs_var   ==> (buff_size, steps, env, 4, 84, 84)
            # out ==> (steps, env/sample_size,...)
            out[:,i] = self.obs_var[idx[i],:, envx[i]]

        return out

    def sample(self, idx):
        '''
            returns samples for a given idx
        '''

        # if sample size > nenv, then adjust the envx to be [0-nenv]
        envx = np.arange(self.batch_size) % self.nenv

        #####
        # define sampling function
        #####
        take = lambda x: self.take(x, idx, envx)

        #####
        # Sampled data.
        #####
        data = {}
        data['actions'] = take(self.actions).as_in_context(self.device) #[nsteps, nenv/sample_size]
        data['rewards'] = take(self.rewards).as_in_context(self.device) #[nsteps, nenv/sample_size]
        data['masks'] = take(self.masks).as_in_context(self.device) #[nsteps, nenv/sample_size]

        ### steps to handle beta after get
        # reshape  #[nsteps, nenv, num_actions] ==> [nsteps * nenv, num_actions]
        data['b_probs'] = take(self.b_probs).as_in_context(self.device).reshape(self.nsteps * self.batch_size, -1)
        data['b_logprobs'] = take(self.b_logprobs).as_in_context(self.device).reshape(self.nsteps * self.batch_size, -1)

        ### steps to handle obs after get
        # 1. slice it slice ==> size: [nsteps, nenv/sample_size,..]
        # a. last_obs = obs[-1]
        # b. obs = obs[-1]
        # 2. reshape (nsteps * num_env, *obs_var.shape[2:])
        #obs = take(self.obs_var).to(self.device) #[nsteps, nenv/sample_size,..]
        #data['last_obs'] = obs[-1] # size: [nenv/sample_size,..]
        #data['obs'] = obs[:-1].view(self.nsteps * self.nenv, *obs.shape[2:])

        obs = mx.nd.array(self.take_numpy(idx, envx)).as_in_context(self.device) #[nsteps, nenv/sample_size,..]
        data['last_obs'] = obs[-1] # size: [nenv/sample_size,..]
        data['obs'] = obs[:-1].reshape(self.nsteps * self.batch_size, *obs.shape[2:])

        return data

    def get(self):
        '''
            Outputs:
            # obs [nsteps + 1, nenv, ...]
            # actions, rewards, dones [nsteps, nenv]

        '''
        assert self.can_sample()

        #####
        # Generate samples ids.
        #####
        idx = np.random.randint(low = 0, high = self.num_in_buffer, size = self.batch_size)
        # if sample size > nenv, then adjust the envx to be [0-nenv]

        return self.sample(idx)
