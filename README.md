# RL control of a MOT based on Coach

This is a fork of the [Coach framework by Intel](https://github.com/IntelLabs/coach), used to control a simulated magneto-optical trap(MOT) through reinforcement learning. Original README.md can be found [here](README_coach.md).


## Table of Contents

- [Installation](#installation)
  * [Docker image](#docker-image)
  * [Manual installation](#manual-installation)

- [Getting Started](#getting-started)
  * [Typical Usage](#typical-usage)
    * [Running Coach](#running-coach) 
    * [Useful options](#useful-options) 
    * [Continue training from a checkpoint](#continue-training-from-a-checkpoint) 
    * [Evaluate only](#evaluate-only) 
  * [Supported Algorithms](#supported-algorithms)


## Installation

Note: Some parts of the original installation that are not needed for MOT control (e.g. PyGame and Gym) have been excluded here.

### Docker image

The corresponding docker images are based on Ubuntu 22.04 with Python 3.7.14. 

We highly recommend starting with the docker image. 

Instructions for the installation of the Docker Engine can be found [here for Ubuntu](https://docs.docker.com/engine/install/ubuntu/)  and [here for Windows](https://docs.docker.com/desktop/install/windows-install/).

Instruction for building of a docker container are [here](docker/README.md).


### Manual installation 


Alternatively, coach can be installed step-by-step:

There are a few prerequisites required. This will setup all the basics needed to get the user going with running Coach:

```
# General
sudo  apt-get update
sudo  apt-get install python3-pip cmake zlib1g-dev python3-tk -y

# Boost libraries
sudo  apt-get install libboost-all-dev -y

# Scipy requirements
sudo  apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran -y

# Other
sudo apt-get install dpkg-dev build-essential libjpeg-dev  libtiff-dev libnotify-dev -y
sudo apt-get install ffmpeg swig curl software-properties-common  build-essential  nasm tar libbz2-dev libgtk2.0-dev  git unzip wget -y

# Python 3.7.14

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.7 python3.7-dev python3.7-venv

# pip

sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
sudo chmod +x /usr/local/bin/patchelf

```

We recommend installing coach in a virtualenv:

```
python3.7 -m venv --copies venv
. venv/bin/activate

```

Clone the repository:
```
git clone https://github.com/MPI-IS/RL-coach-for-MOT.git
```

Install  from the cloned repository:
```
cd RL-coach-for-MOT
pip install .
```


## Getting Started

### Tutorials and Documentation of the original Coach: 

[Jupyter notebooks demonstrating how to run Coach from command line or as a library, implement an algorithm, or integrate an environment](https://github.com/IntelLabs/coach/tree/master/tutorials).

[Framework documentation, algorithm description and instructions on how to contribute a new agent/environment](https://intellabs.github.io/coach/).

### Typical Usage

#### Running Coach

To allow reproducing results in Coach, we use a mechanism called _preset_. 
Several presets can be defined in the `presets` directory.
To list all the available presets use the `-l` flag.

To run a preset, use:

```bash
coach -p <preset_name>
```

For example:
* MOT simulation with continuous control parameters using deep deterministic policy gradients algorithm (DDPG):

```bash
coach -p ContMOT_DDPG
```
  
#### Useful options 

There are several options that are recommended: 
* The `-e` flag allows you to specify the name of the experiment and the folder where the results, logs, and copies of the preset and environment files will be written to. When using the docker container use the `/checkpoint/<experiment name>` -folder to make the results available outside of the container (mounted to `/tmp/checkpoint`). 

* The `-dg` flag enables the output of npz-files containing the output of evaluation episodes to the `npz` folder inside the experiment folder.

* The `-s` flag specifies in seconds the interval at which checkpoints are saved 

For example:

```bash
coach -p ContMOT_DDPG -dg -e /checkpoint/Test -s 1800
```

New presets can be created for different sets of parameters or environments by following the same pattern as in [ContMOT_DDPG](rl_coach/presets/ContMOT_DDPG.py).

Another posibility to change the value of certain parameters is by using the custom parameter flag `-cp`.

For example:

```bash
coach -p ContMOT_DDPG -dg -e /checkpoint/Test -s 1800 -cp "agent_params.exploration.sigma = 0.2"
```

#### Continue training from a checkpoint

The training can be started from an existing checkpoint by specifying its location using the `-crd` flag, this will load the last checkpoint in the folder.

For example: 

```bash
coach -p ContMOT_DDPG -dg -e /checkpoint/Test -s 1800 -crd /checkpoint/Test/18_04_2023-15_20/checkpoint
```


#### Evaluate only


Finally, in order to evaluate the performance of a trained agent without further training use the `--evaluate` flag followed by the number of evaluation steps/episodes?


```bash
coach -p ContMOT_DDPG -dg -e /checkpoint/Test -s 1800 -crd /checkpoint/Test/18_04_2023-15_20/checkpoint --evaluate 10000
```


## Supported Algorithms

<img src="docs_raw/source/_static/img/algorithms.png" alt="Coach Design" style="width: 800px;"/>




### Value Optimization Agents
* [Deep Q Network (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  ([code](rl_coach/agents/dqn_agent.py))
* [Double Deep Q Network (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)  ([code](rl_coach/agents/ddqn_agent.py))
* [Dueling Q Network](https://arxiv.org/abs/1511.06581)
* [Mixed Monte Carlo (MMC)](https://arxiv.org/abs/1703.01310)  ([code](rl_coach/agents/mmc_agent.py))
* [Persistent Advantage Learning (PAL)](https://arxiv.org/abs/1512.04860)  ([code](rl_coach/agents/pal_agent.py))
* [Categorical Deep Q Network (C51)](https://arxiv.org/abs/1707.06887)  ([code](rl_coach/agents/categorical_dqn_agent.py))
* [Quantile Regression Deep Q Network (QR-DQN)](https://arxiv.org/pdf/1710.10044v1.pdf)  ([code](rl_coach/agents/qr_dqn_agent.py))
* [N-Step Q Learning](https://arxiv.org/abs/1602.01783) | **Multi Worker Single Node**  ([code](rl_coach/agents/n_step_q_agent.py))
* [Neural Episodic Control (NEC)](https://arxiv.org/abs/1703.01988)  ([code](rl_coach/agents/nec_agent.py))
* [Normalized Advantage Functions (NAF)](https://arxiv.org/abs/1603.00748.pdf) | **Multi Worker Single Node**  ([code](rl_coach/agents/naf_agent.py))
* [Rainbow](https://arxiv.org/abs/1710.02298)  ([code](rl_coach/agents/rainbow_dqn_agent.py))

### Policy Optimization Agents
* [Policy Gradients (PG)](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) | **Multi Worker Single Node**  ([code](rl_coach/agents/policy_gradients_agent.py))
* [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/abs/1602.01783) | **Multi Worker Single Node**  ([code](rl_coach/agents/actor_critic_agent.py))
* [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) | **Multi Worker Single Node**  ([code](rl_coach/agents/ddpg_agent.py))
* [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)  ([code](rl_coach/agents/ppo_agent.py))
* [Clipped Proximal Policy Optimization (CPPO)](https://arxiv.org/pdf/1707.06347.pdf) | **Multi Worker Single Node**  ([code](rl_coach/agents/clipped_ppo_agent.py))
* [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438) ([code](rl_coach/agents/actor_critic_agent.py#L86))
* [Sample Efficient Actor-Critic with Experience Replay (ACER)](https://arxiv.org/abs/1611.01224) | **Multi Worker Single Node**  ([code](rl_coach/agents/acer_agent.py))
* [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) ([code](rl_coach/agents/soft_actor_critic_agent.py))
* [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://arxiv.org/pdf/1802.09477.pdf) ([code](rl_coach/agents/td3_agent.py))

### General Agents
* [Direct Future Prediction (DFP)](https://arxiv.org/abs/1611.01779) | **Multi Worker Single Node**  ([code](rl_coach/agents/dfp_agent.py))

### Imitation Learning Agents
* Behavioral Cloning (BC)  ([code](rl_coach/agents/bc_agent.py))
* [Conditional Imitation Learning](https://arxiv.org/abs/1710.02410) ([code](rl_coach/agents/cil_agent.py))

### Hierarchical Reinforcement Learning Agents
* [Hierarchical Actor Critic (HAC)](https://arxiv.org/abs/1712.00948.pdf) ([code](rl_coach/agents/hac_ddpg_agent.py))

### Memory Types
* [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495.pdf) ([code](rl_coach/memories/episodic/episodic_hindsight_experience_replay.py))
* [Prioritized Experience Replay (PER)](https://arxiv.org/abs/1511.05952) ([code](rl_coach/memories/non_episodic/prioritized_experience_replay.py))

### Exploration Techniques
* E-Greedy ([code](rl_coach/exploration_policies/e_greedy.py))
* Boltzmann ([code](rl_coach/exploration_policies/boltzmann.py))
* Ornstein–Uhlenbeck process ([code](rl_coach/exploration_policies/ou_process.py))
* Normal Noise ([code](rl_coach/exploration_policies/additive_noise.py))
* Truncated Normal Noise ([code](rl_coach/exploration_policies/truncated_normal.py))
* [Bootstrapped Deep Q Network](https://arxiv.org/abs/1602.04621)  ([code](rl_coach/agents/bootstrapped_dqn_agent.py))
* [UCB Exploration via Q-Ensembles (UCB)](https://arxiv.org/abs/1706.01502) ([code](rl_coach/exploration_policies/ucb.py))
* [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) ([code](rl_coach/exploration_policies/parameter_noise.py))

## Citation

If you used Coach for your work, please use the following citation:

```
@misc{caspi_itai_2017_1134899,
  author       = {Caspi, Itai and
                  Leibovich, Gal and
                  Novik, Gal and
                  Endrawis, Shadi},
  title        = {Reinforcement Learning Coach},
  month        = dec,
  year         = 2017,
  doi          = {10.5281/zenodo.1134899},
  url          = {https://doi.org/10.5281/zenodo.1134899}
}
```

## Contact

We'd be happy to get any questions or suggestions, we can be contacted over [email](mailto:valentin.volchkov@tuebingen.mpg.de)


## Disclaimer

RL-coach-for-MOT is released as a reference code for research purposes. 
