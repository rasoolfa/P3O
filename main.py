import argparse
import mxnet as mx
import os
import time
import numpy as np
from collections import deque
import random
import misc.env as env_fn
from oailibs.common.vec_env.vec_frame_stack import VecFrameStack
from oailibs.common import set_global_seeds
from misc.utils import create_dir, explained_variance, EpisodeStats, csv_writer, safemean
from misc.mxnet_utility import update_linear_schedule
from misc.runner import Runner
from oailibs import logger

parser = argparse.ArgumentParser()

# Optim params
parser.add_argument('--lr', type=float, default=7e-4, help = 'Learning rate')
parser.add_argument('--use_linear_lr_decay', action='store_true', default=False, help='use a linear schedule on the learning rate')
parser.add_argument('--eps', type=float, default=1e-5, help = 'RMSProp epsilon')
parser.add_argument('--alpha', type=float, default=0.99, help = 'RMSProp decay parameter ')

# A2C/P3O params
parser.add_argument('--vf_coef', type=float, default=0.5, help = 'Coefficient of value function loss in the total loss function')
parser.add_argument('--ent_coef', type=float, default=0.01, help = 'Coeffictiant of the policy entropy')
parser.add_argument('--max_gradient_norm', type=float, default=0.5, help = 'Max gradient norm')
parser.add_argument('--tau', type=float, default=0.95 , help = 'Coefficient of gae')
parser.add_argument('--use_gae', default=False, action='store_true', help = 'whether to use GAE or not [default False]')
parser.add_argument('--kl_coef', type=float, default = 0.10, help = 'coefficient of KL')
parser.add_argument('--is_factor', type=float, default = 1.0, help = 'importance weight clipping factor')
parser.add_argument('--frames_waits', type=int, default = 15000, help = 'Min frames to start sampling from replay')
parser.add_argument('--replay_size', type=int, default = 50000, help ='Replay buffer size int(50000)')
parser.add_argument('--replay_ratio', type=int, default = 2, help ='How many (on average) batches of data to sample from the replay buffer')
parser.add_argument('--use_offpolicy_ent', default=False, action='store_true', help ='Entropy for off policy [default False]')
parser.add_argument('--use_ess_is_clipping', default=False, action='store_true', help ='Use ESS for IS clipping[default False]')
parser.add_argument('--sample_mult', type=int, default = 6, help ='is used to decide batch size of off-policies :sample_mult * num_env ')

# RL generic
parser.add_argument('--gamma', type=float, default=0.99, help = 'Discount factor [0,1]')

# General params
parser.add_argument('--hidden_sizes', nargs='+', default= [32, 32])
parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4')
parser.add_argument('--num_env', type=int, default=16)
parser.add_argument('--reward_scale', type=float, default=1.0)
parser.add_argument('--gamestate', default=None)
parser.add_argument('--num_steps', type=int, default=16, help ='number of forward steps')
parser.add_argument('--total_timesteps', type=int, default=int(80e6), help='total number of timesteps to train on ')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--alg_name', type=str, default='a2c', help='can be either a2c or p3o')
parser.add_argument('--disable_cuda', default=False, action='store_true')

parser.add_argument('--log_id', default='dummy')
parser.add_argument('--check_point_dir', default='./ck')
parser.add_argument('--log_dir', default='./log_dir')
parser.add_argument('--log_interval', type=int, default=50, help='log interval, one log per n updates')
parser.add_argument('--save_freq', type=int, default = 500)

parser.add_argument('--save_video_interval', type=int, default = 0, help ='Video save interval')
parser.add_argument('--save_video_length', type=int, default = 100, help ='Video length')

def setup_logAndCheckpoints(args):

    # create folder if not there
    create_dir(args.check_point_dir)

    # log dir for off-policy logs
    create_dir(args.log_dir + '_off_policy')

    fname = str.lower(args.env_name) + '_' + args.alg_name + '_' + args.log_id
    fname_log = os.path.join(args.log_dir, fname)
    fname_offpolicy = os.path.join(args.log_dir + '_off_policy', fname + '_off.csv')
    fname_alt_reward = os.path.join(fname_log, 'stats.csv')

    return os.path.join(args.check_point_dir, fname), fname_log, fname_offpolicy, fname_alt_reward

def write_off_policy_log(wrt_csv_offpolicy, off_policy_stats):

    ##########
    # write off_policy printout in a cvs file
    ##########
    off_policy_stats['returns'] = np.mean(off_policy_stats['returns'])
    del off_policy_stats['adv_per_env']
    del off_policy_stats['policy_entropy']

    if wrt_csv_offpolicy == None:

        wrt_csv_offpolicy = csv_writer(fname_csv_offpolicy, off_policy_stats.keys())
        wrt_csv_offpolicy.writerow(off_policy_stats)

    else:
        wrt_csv_offpolicy.writerow(off_policy_stats)

    return wrt_csv_offpolicy

if __name__ == "__main__":

    args = parser.parse_args()
    print('------------')
    print(args.__dict__)
    print('------------')

    ##############################
    #### Generic setups
    ##############################
    CUDA_AVAL = len(mx.test_utils.list_gpus()) > 0

    if not args.disable_cuda and CUDA_AVAL:
        device = mx.gpu(0)
        print("**** Yayy we use GPU ****")

    else:
        device = mx.cpu()
        print("**** No GPU detected or GPU usage is disabled, sorry! ****")

    #if not args.disable_cuda and CUDA_AVAL and args.cuda_deterministic:
    # please manually set the following environment variables in the terminal
    # `export MXNET_CUDNN_AUTOTUNE_DEFAULT=0`
    # `export MXNET_ENFORCE_DETERMINISM=1`

    # Set seed the RNG for all devices (both CPU and CUDA)
    #mx.random.seed(args.seed)
    #np.random.seed(args.seed)

    ####
    # train and evalution checkpoints, log folders, ck file names
    create_dir(args.log_dir, cleanup = True)

    # create folder for save checkpoints
    ck_fname_part, log_file_dir, fname_csv_offpolicy, fname_alt_reward = setup_logAndCheckpoints(args)
    logger.configure(dir = log_file_dir)
    wrt_csv_offpolicy = None

    ##############################
    #### Init env, model, alg, batch generator etc
    #### Step 1: build env
    #### Step 2: Build model
    #### Step 3: Initiate Alg e.g. a2c
    #### Step 4: Initiate batch/rollout generator
    ##############################

    ##### env setup
    env = env_fn.build_env(seed = args.seed,
                           alg = args.alg_name,
                           env_name = args.env_name,
                           num_env = args.num_env,
                           reward_scale = args.reward_scale,
                           gamestate = args.gamestate,
                           save_video_interval = args.save_video_interval,
                           video_length = args.save_video_length,
                           video_dir = os.path.join(logger.Logger.CURRENT.dir, "videos")
                           )
    if not isinstance(env, VecFrameStack):
        env = VecFrameStack(env, 1)

    ######### SEED ##########
    #  build_env already calls set seed,
    # Set seed the RNG for all devices (both CPU and CUDA)
    set_global_seeds(args.seed)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ##### network setup
    if len(env.observation_space.shape) == 3:
        import models.networks as net
        model = net.CNN(action_space = env.action_space, input_dim = env.observation_space.shape)
        model.initialize(ctx=device)

    elif len(env.observation_space.shape) ==1:
        import models.networks as net
        model = net.MLPBase(action_space = env.action_space, input_dim = env.observation_space.shape)
        model.initialize(ctx=device)

    else:
        raise ValueError("Env %s is not supported" % (env.observation_space.shape))

    print('-----------------------------')
    print("Network Architecture")
    print(model)
    print('-----------------------------')
    print("Name of env:", args.env_name)
    print("Observation_space:", env.observation_space )
    print("Action space:", env.action_space )
    print("Env nstack:", env.nstack )
    print("Env num_envs:", env.num_envs )
    print("Number of steps:", args.num_steps)
    print('----------------------------')

    ##### rollout/batch generator
    rollouts = Runner(env=env, model=model, nsteps=args.num_steps, device=device)
    replay_buffer = None

    ##### algorithm setup
    if str.lower(args.alg_name) == 'a2c':
        import algs.A2C.a2c as alg
        alg = alg.A2C(model= model, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
                      max_gradient_norm=args.max_gradient_norm,
                      gamma=args.gamma,
                      use_gae=args.use_gae,
                      tau=args.tau,
                      device = device)

    elif str.lower(args.alg_name) == 'p3o':
        import algs.P3O.p3o as alg
        alg = alg.P3O(model= model, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
                      max_gradient_norm=args.max_gradient_norm,
                      gamma=args.gamma,
                      use_gae=args.use_gae,
                      tau=args.tau,
                      kl_coef = args.kl_coef,
                      is_factor = args.is_factor,
                      use_offpolicy_ent = args.use_offpolicy_ent,
                      use_ess_is_clipping = args.use_ess_is_clipping,
                      device = device,
                      action_space_type = env.action_space.__class__.__name__
                      )
        from algs.P3O.buffer import Buffer
        replay_buffer = Buffer(num_envs = args.num_env,
                               nsteps=args.num_steps,
                               sample_multiplier = args.sample_mult,
                               frames_waits = args.frames_waits,
                               init_size=args.replay_size)

    else:
        raise ValueError("%s alg is not supported" % args.alg_name)


    ##############################
    # Train and eval
    #############################
    # Calculate the batch_size
    nbatch = args.num_env * args.num_steps
    num_updates = args.total_timesteps // nbatch + 1

    # episode_stats for raw rewards
    episode_stats = EpisodeStats(args.num_steps, args.num_env)
    epinfobuf = deque(maxlen=100)

    # Alternate episode rewards can be used instead of monitors
    alt_eps_num_timesteps = 0
    # Start total timer
    tstart = time.time()

    from misc.optimizer import RMSpropTorch
    optimizer_params = {'learning_rate': args.lr, 'epsilon': args.eps, 'gamma1': args.alpha, 'gamma2': 0}
    optimizer = mx.optimizer.Optimizer.create_optimizer('RMSpropTorch', **optimizer_params)
    trainer = mx.gluon.Trainer(model.collect_params(), optimizer)
    for update in range(1, num_updates):

        #######
        # get one on-policy rollout for all envs
        #######

        with mx.autograd.record():
            data = rollouts.run()

            if replay_buffer: # will be used only if an algs needs it
                replay_buffer.put(obs_var = data['obs'], actions =data['actions'],
                                  b_probs = data['pi_probs'], b_logprobs =data['pi_logs'],
                                  rewards = data['rewards'], masks = data['masks'])

            #######
            # run training that to calculate loss, run backward, and update params
            #######

            # update lr
            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                update_linear_schedule(trainer, update, num_updates, args.lr)

            # train, on-policy steps
            on_policy_stats = alg.train(masks=data['masks'],
                                        rewards=data['rewards'],
                                        values=data['values'],
                                        log_probs=data['logprobs'],
                                        entropies=data['entropies'],
                                        last_values=data['last_values'])

        # clips the gradients
        if str.lower(args.alg_name) == 'a2c' or replay_buffer == None or \
            np.mean(on_policy_stats['returns']):

            params_nd = [x.grad() for x in model.collect_params().values()]
            total_grad_norm = mx.gluon.utils.clip_global_norm(params_nd, args.max_gradient_norm)
            trainer.step(1)
        else:
            total_grad_norm = 0

        on_policy_stats['total_grad_norm'] = total_grad_norm

        if replay_buffer and replay_buffer.has_enough_frames():

            # off policy training
            for _ in range(np.random.poisson(args.replay_ratio)):

                with mx.autograd.record():
                    off_policy_stats = alg.train(replay_buffer = replay_buffer.get())

                params_nd = [x.grad() for x in model.collect_params().values()]
                total_grad_norm = mx.gluon.utils.clip_global_norm(params_nd, args.max_gradient_norm)
                trainer.step(1)
                off_policy_stats['total_grad_norm'] = total_grad_norm

                ####
                # Temp: need to combine off-policy logs (i.e. loss, policy loss)
                # with on-policy log. The following is NOT a good idea
                ####
                if (update % args.log_interval == 0 or update == 1):
                    print('--------OFF-Policy-----------')
                    print("\rpolicy_loss: %.4f \n\rvalue_loss: %.4f\n \
                           \rkl_loss: %.4f\n\rTotal loss: %.4f\n\rtotal_grad_norm: %.4f\n\rreturn [mean]: %.4f" %
                         (off_policy_stats['policy_loss'],
                         off_policy_stats['value_loss'],
                         off_policy_stats['kl_loss'],
                         off_policy_stats['loss'],
                         off_policy_stats['total_grad_norm'],
                         np.mean(off_policy_stats['returns']))
                         )
                    print("kl_coef: %.5f" % (alg.kl_coef))
                    print('-----------------------------')
                    wrt_csv_offpolicy = write_off_policy_log(wrt_csv_offpolicy, off_policy_stats)

        #######
        # logging
        #######
        epinfobuf.extend(data['epinfos'])
        episode_stats.feed(data['raw_rewards'], data['raw_masks'])

        nseconds = time.time() - tstart
        # Calculate the fps (frame per second)
        fps = int(( update * nbatch) / nseconds)

        if (update % args.log_interval == 0 or update == 1 ) and len(epinfobuf) > 1:

            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(data['values_np'], on_policy_stats['returns'])
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(on_policy_stats['policy_entropy']))
            logger.record_tabular("value_loss", float(on_policy_stats['value_loss']))
            logger.record_tabular("policy_loss", float(on_policy_stats['policy_loss']))
            logger.record_tabular("mean_episode_reward", episode_stats.mean_reward())
            logger.record_tabular("mean_episode_length", episode_stats.mean_length())
            logger.record_tabular("loss", float(on_policy_stats['loss']))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("total grad norm", float(on_policy_stats['total_grad_norm']))
            logger.record_tabular("return", float(np.mean(on_policy_stats['returns'])))
            logger.record_tabular('eprewmean', float(safemean([epinfo['r'] for epinfo in epinfobuf])))
            logger.record_tabular('eplenmean', float(safemean([epinfo['l'] for epinfo in epinfobuf])))
            logger.dump_tabular()

        if (update % args.save_freq == 0 or update == num_updates - 1):
            model.save_parameters(ck_fname_part + '.params')

