import sys
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import numpy as np

from oailibs.common.vec_env.vec_video_recorder import VecVideoRecorder
from oailibs.common.vec_env.vec_frame_stack import VecFrameStack
from oailibs.common.cmd_util import make_vec_env, make_env
from oailibs.common.vec_env.vec_normalize import VecNormalize

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

def build_env(seed, alg, env_name, num_env,
              reward_scale, gamestate=None,
              save_video_interval = 0,
              video_length = 5000,
              video_dir = ''
              ):
    '''
      Build the env
    '''
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = num_env or ncpu

    env_type, env_id = get_env_type(env_name)

    if env_type in {'atari', 'retro'}:

        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=gamestate, reward_scale=reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:

       flatten_dict_observations = alg not in {'her'}
       env = make_vec_env(env_id, env_type, num_env or 1, seed, reward_scale=reward_scale, flatten_dict_observations=flatten_dict_observations)

       if env_type == 'mujoco':
           env = VecNormalize(env)


    if  save_video_interval != 0:
        env = VecVideoRecorder(env, video_dir,
                                record_video_trigger=lambda x: x % save_video_interval == 0,
                                video_length=video_length)

    return env


def get_env_type(env_id):
    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id