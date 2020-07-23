# DQN Reinforcement learning Project
Modular Deep Q Network project. Final goal is to reach a Rainbow DQN with the ability to toggle any feature 
beyond default DQN. Current working features included below.

Start training with `python main.py`. If using MineRL, changing command to `xvfb-run python main.py` for training
is recommended. To use expert data, set MINERL_DATA_ROOT environment variable to the location.

Environments currently available for training:
 - MineRLTreechop-v0 (default)
 - Pong


### Features
- [x] DQN
- [x] Double DQN (Default is active, disable with `--no-double-dqn`)
- [x] Dueling DQN (Default is active, disable with `--no-dueling`)
- [x] n-step DQN (`--n-step N`, where `N=1` is the normal case)
- [x] Prioritized Experience Replay (Default is active, disable with `--no-prioritized`)
  - [x] Alternate uniform sampling memory (`--no-prioritized`)
- [x] Noisy Networks for Exploration (`--noisy`, use with `--greedy` for no e-greedy) 
(credit to  [this repository](https://github.com/Kaixhin/Rainbow))
- [x] C51 (`--use-c51`) (credit to [this repository](https://github.com/Kaixhin/Rainbow))
- [x] Continuous -> Discrete
- [x] Action branching (active by default)
  - [ ] No branching option (Not yet fully tested, use with `--no-action-branching`)
- [x] MineRL Treechop
- [x] DQfD
- [x] Logging
  - [x] Enable `wandb` logging (`--log-run`, combine with `--verbosity 0` to avoid large terminal upload.)
  - [x] Enable local csv logging (`--local-log`, stored in the [runs_recordings](/runs_recordings) dir.)

### Results
![Mean plot](resources/mean_plot_20_ep.png)
<img src="resources/mean_plot_20_ep.png" width="800" height="500" />
![Median plot](resources/median_plot_20_ep.png)

### Sample video
Click [here](https://drive.google.com/file/d/1UrMakspToYwSogeae1UkFy_On-lfspg9/view?usp=sharing) to see a better quality version at 1x speed.

![Sample video](resources/Treechop_vid.gif)
