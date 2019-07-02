from __future__ import  print_function, division
import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.loss import Loss
import numpy as np

class P3O:

    def __init__(self, model,
                 vf_coef=0.5,
                 ent_coef=0.01,
                 max_gradient_norm=0.5,
                 gamma=0.99,
                 use_gae=False,
                 tau=0.95,
                 kl_coef=0.5,
                 max_kl_coef=0.6,
                 is_factor=1,
                 use_offpolicy_ent=False,
                 use_ess_is_clipping=False,
                 device='cpu',
                 action_space_type='Discrete'
                 ):
        '''
            model: policy network architecture.
            seed:  seed to make random number sequence in the alorightm reproducible.
            vf_coef:  coefficient of value function loss in the total loss function.
            ent_coef: coeffictiant of the policy entropy in the total loss function.
            max_gradient_norm:  gradient is clipped to have global L2 norm no more than this value
            lr:   learning rate for RMSProp
            eps:  RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update)
            alpha: RMSProp decay parameter
            gamma: reward discounting parameter
            use_gae: use whether or not Generalized Advantage Estimation
            tau:  coefficient of gae
            is_factor: importance weight clipping factor
            kl_coef: coefficient of KL
            max_kl_coef: max KL coef that can go
            use_offpolicy_ent: if it is False, there would be no entropy maximization for off policy
            use_ess_is_clipping: use ESS for computing lambda as clipping variable
            device: to indicate use of cpu or gpu
        '''
        self.model = model
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_gradient_norm = max_gradient_norm
        self.gamma = gamma
        self.tau = tau
        self.use_gae = use_gae
        self.is_factor = is_factor
        self.initial_is_factor = is_factor
        self.device = device
        self.kl_loss   = mx.gluon.loss.KLDivLoss(axis=0)
        self.use_offpolicy_ent = use_offpolicy_ent
        self.use_ess_is_clipping = use_ess_is_clipping

        ## KL params
        self.kl_coef = kl_coef
        self.max_kl_coef = max_kl_coef
        self.initial_kl_coef = kl_coef
        self.kl_coef_increase_step = 250
        self.r_eps = np.float32(1e-7) # this is used to avoid inf or nan in calculations

        if action_space_type == 'Box':
            self.continous_action_space = True

        elif action_space_type == 'Discrete':
            self.continous_action_space = False

        else:
            raise NotImplementedError

    def update_kl_IS_coefs_by_ESS(self, w):
        '''
            This function calculates effective sample size (ESS):
            ESS = ||w||^2_1 / ||w||^2_2  , w = pi / beta
            ESS = ESS / n where n is number of samples to normalize
            pi: is target dist (n_sample * n_steps, 1)
            beta: is behavioral/proposal dist (n_sample * n_steps, 1)
            w: is (n_sample * n_steps, 1)
        '''
        n = w.shape[0]
        w = ( (mx.nd.sum(w)**2) / (mx.nd.sum(w**2)) + self.r_eps) / n
        ess_factor = np.float32(np.asscalar(w.asnumpy()))

        coef = 1.0 - ess_factor ## close to zero means the dist are very close,

        if np.isnan(coef) or np.isinf(coef): # make sure that it is valid
            coef = self.initial_kl_coef
            ess_factor = self.initial_is_factor

        elif coef <= self.r_eps: # to avoid numerical problem
            coef = self.r_eps
            ess_factor = self.r_eps

        self.kl_coef = np.float32(coef)

        # update the important sampling clip with ess if it is on
        if self.use_ess_is_clipping == True:
            self.is_factor = np.float32(ess_factor)


    def train(self, masks = None, rewards= None, values= None, log_probs= None,
              entropies= None, last_values= None, replay_buffer = None):
        '''
            Note: when replay_buffer is NOT empty, all inputs will be ignored as they will be
            fetched from replay buffer
            inputs:
                masks :  [num_steps + 1, num_env]
                rewards: [num_steps, num_env]
                values: [num_steps, num_env, 1]
                log_probs: [num_steps, num_env]
                entropies: [num_steps, num_env]
                last_values: [num_env, 1]
            outputs:
                val_loss, policy_loss, mean(entropies.mean)
        '''
        kl_term = 0.0
        kl_term_out = 0.0
        ratio_stats = 0
        if replay_buffer:

            #######
            # Get data from replay
            #######
            masks = replay_buffer['masks']     #[num_steps + 1, num_env]
            rewards = replay_buffer['rewards'] #[num_steps, num_env]

            if self.continous_action_space == True:
                # Will add later
                raise NotImplementedError
            else:
                beta_actions = replay_buffer['actions'] # [nsteps, num_env]
                beta_probs = replay_buffer['b_probs'] #  [nsteps * num_env, num_actions]
                beta_logs = replay_buffer['b_logprobs'] # [nsteps * num_env, num_actions]

                #######
                # Feed data to the model
                #######

                # obs_var is reshaped to be nsteps * num_env, *obs_var.shape[2:]
                entropies, values, _, pi_probs_logs = self.model(replay_buffer['obs'].as_in_context(self.device), None, False)
                _, last_values, _, _ = self.model(replay_buffer['last_obs'].as_in_context(self.device), None, False) # [num_env, 1]

                num_steps, num_envs = rewards.shape
                entropies = entropies.reshape(num_steps, num_envs)
                values = values.reshape(num_steps, num_envs, 1)  # num_steps * num_env, 1 ==> [num_steps, num_env, 1]
                pi_probs = pi_probs_logs[0] # probs [nsteps * num_env, num_actions]
                pi_logs  = pi_probs_logs[1] # logs [nsteps * num_env, num_actions]

                #######
                # get pi_[logs/probs](at|st) where at ~ beta.  ==> off-policy trajectory
                # get beta_[logs/probs](at|st) where at ~ beta ==> on-policy trajectory
                #######
                ind = mx.nd.stack(mx.nd.arange(beta_actions.reshape(-1, 1).size), beta_actions.reshape(-1, 1).astype('float32')).as_in_context(self.device)
                pi_sampled_probs = mx.nd.gather_nd(pi_probs, indices=ind) #(nsteps * num_env, 1)
                log_probs  = mx.nd.gather_nd(pi_logs, indices=ind)  #(nsteps * num_env, 1)
                beta_sampled_probs = mx.nd.gather_nd(beta_probs.as_in_context(self.device), indices=ind) #(nsteps * num_env, 1)

                #######
                # compute kl_loss(beta|pi) where bi is target dist
                #######
                kl_term = self.kl_loss(beta_logs.detach().as_in_context(self.device), pi_probs).mean()
                kl_term_out = kl_term.asscalar()


            #######
            # compute importance sampling AND ratio rho = min(c,pi/beta)
            #######
            ratio = pi_sampled_probs.detach() / (beta_sampled_probs + self.r_eps)

            # Update KL coef and IS factor by ESS
            self.update_kl_IS_coefs_by_ESS(ratio)

            # IS clipping
            rho = mx.nd.clip(ratio, 0, self.is_factor)
            rho = rho.reshape(num_steps, num_envs, 1) # (nsteps * num_env, 1) ==> (nsteps, num_env, 1)
            ratio_stats = ratio.asnumpy().mean()

            assert not np.isnan(ratio_stats) and not np.isinf(ratio_stats), ('ratio is nan or inf %.4f' % ratio_stats)

            # disable entropy during for off policy step
            if self.use_offpolicy_ent == False:

                entropies = entropies * 0.0

        ######
        # Define/resize some of the tensors
        ######
        num_steps, num_envs = rewards.shape
        returns = mx.nd.zeros((num_steps + 1 , num_envs, 1)).as_in_context(self.device)
        mu_ent = entropies.mean().as_in_context(self.device)
        rewards = rewards.reshape(num_steps , num_envs, 1).as_in_context(self.device)
        log_probs   = log_probs.reshape(num_steps , num_envs, 1).as_in_context(self.device) #num_steps , num_envs ==> (num_steps , num_envs, 1)
        masks   = masks.reshape(num_steps + 1 , num_envs, 1).as_in_context(self.device)
        value_loss = 0.0
        policy_loss = 0.0
        gae = 0.0

        #######
        #calculate discounted returns or gae
        #######
        if self.use_gae:

            returns = returns.detach()
            values_detach = values.detach()
            masks_detach = masks.detach()
            rewards_detach = rewards.detach()
            # Generalized Advantage Estimataion
            for step in reversed(range(num_steps)):

                if (step == num_steps - 1): # since the gae requires last values and it is in a separate var
                    delta_t = rewards_detach[step] + (1.0 - masks_detach[step + 1]) * self.gamma * last_values.detach() - values_detach[step]

                else:
                    delta_t = rewards_detach[step] + (1.0 - masks_detach[step + 1]) * self.gamma * values_detach[step + 1] - values_detach[step]

                gae = delta_t + (1.0 - masks_detach[step + 1]) * self.gamma * self.tau * gae
                returns[step] = gae + values_detach[step]

        else:
            # discounted returns
            returns[-1] = last_values.detach()
            for step in reversed(range(num_steps)):
                    returns[step] = (1.0 - masks[step + 1]) * self.gamma * returns[step + 1] + rewards[step]

        #######
        # get the Adv.
        #######
        advantages = returns[:-1] - values  #[num_steps, num_env, 1]

        #######
        # calculate value  and policy losses
        #######
        value_loss = (advantages**2).mean()

        if replay_buffer:
            policy_loss = - (rho.detach() * advantages.detach() * log_probs).mean()

        else:
            policy_loss = - (advantages.detach() * log_probs).mean()

        # loss calculation
        if isinstance(self.kl_coef, np.ndarray):
            self.kl_coef = self.kl_coef[0]
        loss = value_loss * self.vf_coef + policy_loss - mu_ent * self.ent_coef + self.kl_coef * kl_term

        # num_steps , num_envs ==> (num_envs, num_steps) to be consistent with values_np
        returns_for_show = returns[:-1].asnumpy().swapaxes(1, 0)

        # run backward step
        loss.backward()

        out = {}
        out['value_loss'] = value_loss.asscalar()
        out['policy_loss'] = policy_loss.asscalar()
        out['policy_entropy'] = mu_ent.asscalar()
        out['loss'] = loss.asscalar()
        out['returns'] = returns_for_show.flatten()
        out['kl_loss'] = kl_term_out
        out['ratio_stats'] = ratio_stats
        out['ess_value'] = self.is_factor
        ####
        #  advantage ==> [num_steps, num_env, 1]
        #  adv_per_env contains adv average per env and has (num_env,)
        #  Plus, clip the values to be between 0,1
        #  use this value for replay buffer
        #####
        t_adv_per_env = advantages.detach().asnumpy().mean(axis=0).reshape(-1).tolist()
        out['adv_per_env'] = np.clip(t_adv_per_env, a_min= 0, a_max =1)

        return out
