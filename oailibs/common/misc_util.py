import gym
import numpy as np
import os
import pickle
import random
import tempfile
import zipfile

def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i  + 1000 * rank if i is not None else None

    import mxnet as mx
    mx.random.seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)