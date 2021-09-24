# SimpleSAC
A simple and modular implementation of the [Soft Actor Critic](https://arxiv.org/abs/1812.05905) algorithm in PyTorch.


## Installation

1. Install and use the included Ananconda environment
```
$ conda env create -f environment.yml
$ source activate SimpleSAC
```
You'll need to [get your own MuJoCo key](https://www.roboti.us/license.html) if you want to use MuJoCo.

2. Add this repo directory to your `PYTHONPATH` environment variable.
```
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

## Run Experiments
## Run Experiments
You can run SAC experiments using the following command:
```
python -m SimpleSAC.sac_main \
    --env 'HalfCheetah-v2' \
    --logging.output_dir './experiment_output' \
    --device='cuda'
```
If you want to run on CPU only, just omit the `--device='cuda'` part.
All available command options can be seen in SimpleSAC/sac_main.py and SimpleSAC/sac.py.


## Visualize Experiments
You can visualize the experiment metrics with viskit:
```
python -m viskit './experiment_output'
```
and simply navigate to [http://localhost:5000/](http://localhost:5000/)


## Weights and Biases Online Visualization Integration
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site). To log to W&B, you first need to set your W&B API key environment variable:
```
export WANDB_API_KEY='YOUR W&B API KEY HERE'
```

Then you can run experiments with W&B logging turned on:

```
python -m SimpleSAC.sac \
    --env 'HalfCheetah-v2' \
    --logging.output_dir './experiment_output' \
    --device='cuda' \
    --logging.online
```





## Credits
The project organization is inspired by [TD3](https://github.com/sfujim/TD3).
The SAC implementation is based on [rlkit](https://github.com/vitchyr/rlkit).
The viskit visualization is taken from [viskit](https://github.com/vitchyr/viskit), which is taken from [rllab](https://github.com/rll/rllab).

