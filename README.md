P3O: Policy-on Policy-off Policy Optimization
=============================================

On-policy reinforcement learning (RL) algorithms have high sample complexity while off-policy algorithms are difficult to tune. Merging the two holds the promise to develop efficient algorithms that generalize across diverse environments. It is however challenging in practice to find suitable hyper-parameters that govern this trade off. This paper develops a simple algorithm named P3O that interleaves off-policy updates with on-policy updates. P3O uses the effective sample size between the behavior policy and the target policy to control how far they can be from each other and does not introduce any additional hyper-parameters. Extensive experiments on the Atari-2600 and MuJoCo benchmark suites show that this simple technique is highly effective in reducing the sample complexity of state-of-the-art algorithms.


This repository provides the MXNet implementation of [P3O: Policy-on Policy-off Policy Optimization](https://arxiv.org/abs/1905.01756). If you use this code please cite the paper using the bibtex reference below.

```
@article{fakoorp3o,
  author    = {Rasool Fakoor and
               Pratik Chaudhari and
               Alexander J. Smola},
  title     = {{P3O:} Policy-on Policy-off Policy Optimization},
  journal   = {CoRR},
  volume    = {abs/1905.01756},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.01756},
}

```

# Getting Started

On Ubuntu 16.04, run the following script to install the environment dependencies:

```
sudo apt-get install -y libsm6 libxrender1 libfontconfig1

wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh && \ 
bash Anaconda3-2018.12-Linux-x86_64.sh && \
source ~/.bashrc && conda update -y conda && conda update -y anaconda

conda create -n gluonrl python=3.7.1 anaconda && conda activate gluonrl && \
conda install -y -n gluonrl -c conda-forge pyhamcrest

pip install gym && conda install -y -n gluonrl -c ska pygtk && \
pip install pyopengl opencv-python gym[atari] mxnet
```

Set the following environmental variables for reproducible experiments:

```
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENFORCE_DETERMINISM=1
export OMP_NUM_THREADS=1
```

# Running the code

```
python -u main.py --use_linear_lr_decay --use_ess_is_clipping --frames_waits 15000 --sample_mult 6 --num_steps 16 --num_env 16 --save_freq 500 --log_interval 40 --replay_ratio 2 --replay_size 50000 --log_id log_0 --ent_coef 0.01 --seed 0 --env=BreakoutNoFrameskip-v4 --alg_name p3o --use_gae 
```

'env' can be any of 49 atari games. The codes can be used either on GPU or CPU machine. For the experiments in this paper, we used [c5.18xlarge](https://aws.amazon.com/ec2/instance-types/c5/) .

For complete list of hyperparameters, please refer to the paper appendix. 

# Reporting Issues and Contacting 

Please create an issue to report a problem or to ask question in the repo's issues page or contact [Rasool](https://github.com/rasoolfa).

# Acknowledgement 
- Special thanks to [Hang Zhang](https://github.com/zhanghang1989) and [Tong He](https://github.com/hetong007) for their helps and tireless efforts with MXNet implementation.
- Vectorized environment generation such as Atari and MujoCo, environment wrappers, monitoring, logging, etc are based/copied on/from [OpenAI Baselines](https://github.com/openai/baselines). [p3o/oailibs](https://github.com/zhanghang1989/p3o/tree/master/oailibs) contains related codes to OpenAI Baselines.
