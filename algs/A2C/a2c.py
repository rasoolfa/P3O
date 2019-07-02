from __future__ import  print_function, division
import mxnet as mx

class A2C:

    def __init__(self, model,
                 vf_coef=0.5,
                 ent_coef=0.01,
                 max_gradient_norm=0.5,
                 gamma=0.99,
                 use_gae = False,
                 tau = 0.95,
                 device = 'cpu'
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

        '''

        self.model = model
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_gradient_norm = max_gradient_norm
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.use_gae = use_gae

    def train(self, masks = None, rewards= None, values= None, log_probs= None, entropies= None, last_values= None):
        '''
            inputs:
                masks :  [num_steps + 1, num_env]
                rewards: [num_steps, num_env]
                values: [num_steps, num_env, 1]
                log_probs: [num_steps, num_env]
                entropies: [num_steps, num_env]
            outputs:
                val_loss, policy_loss, mean(entropies.mean)
        '''
        num_steps, num_envs = rewards.shape[0], rewards.shape[1]
        mu_ent = entropies.mean().as_in_context(self.device)
        returns = mx.nd.zeros((num_steps + 1 , num_envs, 1)).as_in_context(self.device)
        rewards = rewards.reshape(num_steps , num_envs, 1).as_in_context(self.device)
        log_probs = log_probs.reshape(num_steps , num_envs, 1).as_in_context(self.device) #num_steps , num_envs ==> (num_steps , num_envs, 1)
        masks   = masks.reshape(num_steps + 1 , num_envs, 1).as_in_context(self.device)
        value_loss = 0.0
        policy_loss = 0.0
        gae = 0.0

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

        # get the Adv.
        advantages = returns[:-1] - values  #[num_steps, num_env, 1]

        # calculate value loss
        value_loss = (advantages**2).mean()
        policy_loss = - (advantages.detach() * log_probs).mean()

        # run backward step
        loss = value_loss * self.vf_coef + policy_loss - mu_ent * self.ent_coef
        loss.backward()

        # num_steps , num_envs ==> (num_envs, num_steps) to be consistent with values_np
        returns_for_show = returns[:-1].asnumpy().swapaxes(1, 0)

        out = {}
        out['value_loss'] = value_loss.asscalar()
        out['policy_loss'] = policy_loss.asscalar()
        out['policy_entropy'] = mu_ent.asscalar()
        out['loss'] = loss.asscalar()
        out['returns'] = returns_for_show.flatten()
        out['kl_loss'] = 0.0

        return out

