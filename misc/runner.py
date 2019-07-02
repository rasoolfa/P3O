import numpy as np
from collections import deque
from mxnet import nd

class Runner:
    """
      This class generates batches of experiences
    """
    def __init__(self, env, model, nsteps=5, device = 'cpu'):
        '''
            nsteps: number of steps 
        '''
        self.env = env
        self.model = model
        #print('Runner self.model', self.model)
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv * nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.init_states() # states are useful for recurrent cases 
        self.dones = [False for _ in range(nenv)]
        self.device = device


    def reset(self):
        """
            This function restart the env
        """
        #self.obs[:] = env.reset()
        self.states = model.init_states()

    def run(self):
        '''
            This function returns mini batch of experiences
        '''
        # We initialize the lists that will contain the mini batch(mb) of experiences
        mb_obs, mb_rewards, mb_actions, mb_logprobs, mb_entropies, mb_values, mb_dones = [],[],[],[],[],[],[]
        mb_all_act_probs, mb_all_act_logits = [], []
        mb_states = self.states
        epinfos = []

        for n in range(self.nsteps):
            
            # Given observations, take action and value (V(s))
            # We already have self.obs because self.obs[:] = env.reset() on init
            if len(self.obs.shape) == 4:
                # if input image ==> (nenv, h,w,c) ==> (nenv, c, h,w) 
                self.obs = self.obs.transpose((0, 3, 1, 2)) #/ 255.0

            obs_var = self.obs.copy()
            actions, logprobs, entropy, values, states, action_probs_logits = self.model(nd.array(obs_var, ctx=self.device), self.states)

            # Append the experiences
            mb_obs.append(np.copy(obs_var)) # mb_obs is(steps, nenv,  c, h, w)
            mb_actions.append(actions) # size is nenv
            mb_entropies.append(entropy)# size is nenv
            mb_logprobs.append(logprobs) # size is nenv * num_actions
            mb_values.append(values) # values is is nenv * 1
            mb_dones.append(self.dones) # self.dones is nenv * 1

            # action_probs_logits ==> (probs, logits)
            mb_all_act_probs.append(action_probs_logits[0])
            mb_all_act_logits.append(action_probs_logits[1])

            # Take actions in env and look the results
            actions = actions.asnumpy()
            obs, rewards, dones, vinfos = self.env.step(actions)
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)


            ####################
            ### collect reward info
            for info in vinfos:
                if 'episode' in info.keys():
                    epinfos.append(info['episode'])
            ###################

        if len(self.obs.shape) == 4:
            # if input image ==> (nenv, h,w,c) ==> (nenv, c, h,w)
            obs_var = self.obs.transpose((0, 3, 1, 2)) #/ 255.0

        else:
            obs_var = self.obs.copy()

        _, _, _, last_values, _, _ = self.model(nd.array(obs_var, ctx=self.device), self.states)
        
        # need to keep track of last step to see whether or not it is terminal 
        mb_obs.append(obs_var)
        mb_dones.append(self.dones)
        
        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_actions = nd.stack(*mb_actions, axis=0) #(nsteps, nenv)
        mb_all_act_probs = nd.stack(*mb_all_act_probs, axis=0) #(nsteps, nenv, num_actions)
        mb_all_act_logits = nd.stack(*mb_all_act_logits, axis=0) #(nsteps, nenv, num_actions)
        mb_values = nd.stack(*mb_values, axis=0) # (nsteps, nenv, 1)
        mb_logprobs =  nd.stack(*mb_logprobs, axis=0) # (nsteps, nenv)
        mb_entropies =  nd.stack(*mb_entropies, axis=0) # (nsteps, nenvs)
        mb_raw_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0) # (nsteps, nenvs)  --> # (nenvs, nsteps)
        mb_rewards = nd.array(np.asarray(mb_rewards, dtype=np.float32)) # (nsteps, nenvs)
        mb_dones = np.asarray(mb_dones, dtype=np.bool) #(nsteps + 1, nenvs)
        mb_masks = nd.array(mb_dones.astype(np.float32)) # (nsteps + 1, nenvs)

        # raw mask for reporting purpose
        mb_raw_masks = mb_dones.swapaxes(1, 0) #  (nenvs, nsteps + 1)
        mb_raw_masks = mb_raw_masks[:, :-1] # (nenvs, nsteps)
        # convert to numpy and remove last values (nsteps, nenvs) --> # (nenvs, nsteps)
        mb_vals_np = mb_values.squeeze(axis=-1).detach().asnumpy().swapaxes(1, 0)

        info = {}
        info['obs'] = mb_obs
        info['states'] = mb_states
        info['rewards'] = mb_rewards
        info['mb_dones'] = mb_dones
        info['masks'] = mb_masks
        info['actions'] = mb_actions
        info['logprobs'] = mb_logprobs
        info['entropies'] = mb_entropies
        info['values'] = mb_values
        info['last_values'] = last_values
        info['values_np'] = mb_vals_np.flatten()
        info['pi_probs'] = mb_all_act_probs
        info['pi_logs'] = mb_all_act_logits
        info['epinfos'] = epinfos

        # raw masks and raw rewards will be used to report results
        info['raw_rewards'] = mb_raw_rewards.flatten()
        info['raw_masks'] = mb_raw_masks.flatten()

        return info
