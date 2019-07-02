from __future__ import  print_function, division
import mxnet as mx
import mxnet.gluon.nn as nn
import mxnet.ndarray as F
import numpy as np
from misc.utils import identity, get_action_info


class Flatten(nn.Block):
    def forward(self, x):
        return x.reshape(0, -1)

def broadcast_tensors(x1, x2):
    outshape = x1.shape[:-1] + (x2.shape[-1],)
    return x1.broadcast_to(outshape), x2.broadcast_to(outshape)

class CategoricalDistribution(object):
    def __init__(self, logits):
        r"""
        Creates a categorical distribution parameterized by either :attr:`probs` or
        :attr:`logits` (but not both).

        Samples are integers from :math:`\{0, \ldots, K-1\}` where `K` is ``probs.size(-1)``.

        Args:
            logits (Tensor): event log probabilities
        """
        # numeric stable way?
        self.logits = mx.nd.log_softmax(logits, axis=-1)
        self.probs = logits.softmax(axis=-1)
        self._batch_shape = self.logits.shape[:-1] if self.logits.ndim > 1 else tuple()
        self._num_events = self.logits.shape[-1]

    def sample(self, sample_shape=tuple()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        sample_shape = self._extended_shape(sample_shape)
        param_shape = sample_shape + (self._num_events,)
        probs = self.probs.broadcast_to(param_shape)
        probs_2d = probs.reshape(-1, self._num_events)
        sample_2d = mx.random.multinomial(probs_2d, 1)
        return sample_2d.reshape(sample_shape)
        

    def log_prob(self, value):
        """
        Argument validation for distribution methods such as `log_prob`,
        `cdf` and `icdf`. The rightmost dimensions of a value to be
        scored via these methods must agree with the distribution's batch
        and event shapes.
        Args:
            value (Tensor): the tensor whose log probability is to be
                computed by the `log_prob` method.
        Raises
            ValueError: when the rightmost dimensions of `value` do not match the
                distribution's batch and event shapes.
        """
        value = value.expand_dims(-1)
        value, log_pmf = broadcast_tensors(value, self.logits)
        ind = mx.nd.stack(mx.nd.arange(value.shape[0], ctx=value.context), value[:, 0].astype('float32'))
        return mx.nd.gather_nd(log_pmf, indices=ind)

    def entropy(self):
        """
        Returns entropy of distribution, batched over batch_shape.
        Returns:
            Tensor of shape batch_shape.
        """
        p_log_p = self.logits * self.probs
        return -p_log_p.sum(-1)
    
    def _extended_shape(self, sample_shape=tuple()):
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).
        Args:
            sample_shape tuple(): the size of the sample to be drawn.
        """
        if not isinstance(sample_shape, tuple):
            sample_shape = tuple(sample_shape)
        return sample_shape + self._batch_shape


class Categorical(nn.Block):
    '''
        This creates categorical dist
    '''
    def __init__(self, num_inputs, num_actions):

        super(Categorical, self).__init__()
        self.out  = nn.Dense(in_units=num_inputs, units=num_actions,
                             weight_initializer=mx.initializer.Orthogonal(0.01),
                             bias_initializer='zeros')

    def forward(self, x, sample_action):
        '''
            x: B * num_inputs
        '''
        actor = self.out(x)
        ###########
        # For policy graident, we need to sample from actions
        # logprob of sampled actions
        # entropy  - prob_all * log_prob_all
        ###########

        cat = CategoricalDistribution(logits = actor)
        entropy = cat.entropy()

        # if action are provided, then no action sampling should happen
        if sample_action == True:
            actions = cat.sample()
            logprobs = cat.log_prob(actions)
            return actions, logprobs, entropy, (cat.probs, cat.logits)
        else:
            return entropy, (cat.probs, cat.logits)


class CNN(nn.Block):
    """
    This creates fully conv layers. 
    This arch is standard nature cnn model.
    """
    def __init__(self, action_space, hidden_sizes = [512], input_dim = None, output_dim = None, 
            hidden_activation=F.relu,
            output_activation=None):

        super(CNN, self).__init__()
        self.hidden_size = hidden_sizes[0]
        self.hid_act = hidden_activation
        self.out_act = output_activation
        self.obs_shape = input_dim # assume input_dim is (H, W, C )
        num_actions, action_space_type = get_action_info(action_space)


        #### build multi-layers CNN
        self.cnn = nn.Sequential()
        self.cnn.add(nn.Conv2D(in_channels = self.obs_shape[-1] , channels = 32, kernel_size = 8, strides=4,
                               weight_initializer=mx.initializer.Orthogonal(1.414),
                               bias_initializer='zeros'))
        self.cnn.add(nn.Activation('relu'))
        self.cnn.add(nn.Conv2D(in_channels = 32, channels = 64, kernel_size = 4, strides=2,
                               weight_initializer=mx.initializer.Orthogonal(1.414),
                               bias_initializer='zeros'))
        self.cnn.add(nn.Activation('relu'))
        self.cnn.add(nn.Conv2D(in_channels = 64, channels = 32, kernel_size = 3, strides=1,
                               weight_initializer=mx.initializer.Orthogonal(1.414),
                               bias_initializer='zeros'))
        self.cnn.add(nn.Activation('relu'))
        self.cnn.add(Flatten())
        self.cnn.add(nn.Dense(in_units=32 * 7 * 7, units=self.hidden_size,
                              weight_initializer=mx.initializer.Orthogonal(1.414),
                              bias_initializer='zeros'))
        self.cnn.add(nn.Activation('relu'))

        ###########
        # critic head
        ###########
        self.critic = nn.Dense(in_units=self.hidden_size, units=1,
                               weight_initializer=mx.initializer.Orthogonal(1.0),
                               bias_initializer='zeros')

        ###########
        # actor head
        ###########
        if action_space_type == 'Discrete': ## for atari game
            self.dist = Categorical(num_inputs = self.hidden_size , num_actions = num_actions)

        elif action_space_type == 'Box': #Mujoco
            raise NotImplementedError

        else:
            raise NotImplementedError

        #self.train()

    def forward(self, x, state=None, sample_action=True):
        '''
            input (x): B * D where B is batch size and D is input_dim
            return: B * S where S is output_dim
            cat.probs: num_env * num_actions
        '''

        # run through the model
        x = x / 255.0
        x = self.cnn(x)
        critic  = self.critic(x)
        
        # if action are provided, then no action sampling should happen
        if sample_action == True:

            actions, logprobs, entropy, probs_logits = self.dist(x, sample_action)
            return actions, logprobs, entropy, critic, state, probs_logits

        else:
            entropy, probs_logits = self.dist(x, sample_action)

            return entropy, critic, state, probs_logits
    
    def init_states(self):
        '''
            This functions is here to support recurrent based model
        '''
        return None


class MLPBase(nn.Block):
    """
      This arch is standard for some of gym games
    """
    def __init__(self, action_space, hidden_sizes = [64], input_dim = None, output_dim = None, 
            hidden_activation=F.relu,
            output_activation=None):
        '''
            input_dim is a tuple, (N,)
        '''

        super(MLPBase, self).__init__()
        self.hidden_size = hidden_sizes[0]
        num_actions, action_space_type = get_action_info(action_space)

        # actor network
        self.actorFC = nn.Sequential()
        self.actorFC.add(nn.Dense(in_units=input_dim[0], units=self.hidden_size,
                                  weight_initializer=mx.initializer.Orthogonal(1.414),
                                  bias_initializer='zeros'))
        self.actorFC.add(nn.Tanh())
        self.actorFC.add(nn.Dense(in_units=self.hidden_size, units=self.hidden_size,
                                  weight_initializer=mx.initializer.Orthogonal(1.414),
                                  bias_initializer='zeros'))
        self.actorFC.add(nn.Tanh())
        # crtic network
        self.criticFC = nn.Sequential()
        self.criticFC.add(nn.Dense(in_units=input_dim[0], units=self.hidden_size,
                                   weight_initializer=mx.initializer.Orthogonal(1.414),
                                   bias_initializer='zeros'))
        self.criticFC.add(nn.Tanh())
        self.criticFC.add(nn.Dense(in_units=self.hidden_size, units=self.hidden_size,
                                   weight_initializer=mx.initializer.Orthogonal(1.414),
                                   bias_initializer='zeros'))
        self.criticFC.add(nn.Tanh())
        #############
        # Critic Head
        #############
        self.critic = nn.Dense(in_units=self.hidden_size, units=1,
                               weight_initializer=mx.initializer.Orthogonal(1.414),
                               bias_initializer='zeros')

        ###########
        # actor head
        ###########
        if action_space_type == 'Discrete': ## for atari game
            self.dist = Categorical(num_inputs = self.hidden_size , num_actions = num_actions)

        elif action_space_type == 'Box': #Mujoco
            raise NotImplementedError

        else:
            raise NotImplementedError

        self.train()

    def forward(self, x, state = None, sample_action = True):
        '''
            input (x): B * D where B is batch size and D is input_dim
            return: B * S where S is output_dim
            cat.probs: num_env * num_actions
        '''

        # run through the model
        critic  = self.critic(self.criticFC(x))
        actor_feats  =  self.actorFC(x)

        # if action are provided, then no action sampling should happen
        if sample_action == True:

            actions, logprobs, entropy, probs_logits = self.dist(actor_feats, sample_action)
            return actions, logprobs, entropy, critic, state, probs_logits

        else:

            entropy, probs_logits = self.dist(actor_feats, sample_action)
            return entropy, critic, state, probs_logits

    def init_states(self):
        '''
            This functions is here to support recurrent based model
        '''
        return None
