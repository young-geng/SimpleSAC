import random
import pprint
import time
import tempfile
import uuid
import os

import numpy as np

import absl.flags
from absl import logging

import torch


class WandBLogger(object):
    def __init__(self, wandb_logging, variant, project, experiment_id=None,
                 prefix='', output_dir=None, random_time=0.0):
        self.wandb_logging = wandb_logging
        if wandb_logging:
            global wandb
            import wandb

            if experiment_id is None:
                experiment_id = uuid.uuid4().hex

            if prefix != '':
                project = '{}--{}'.format(prefix, project)

            if output_dir is None:
                output_dir = tempfile.mkdtemp()
            else:
                output_dir = os.path.join(output_dir, experiment_id)
                os.makedirs(output_dir, exist_ok=True)

            if random_time > 0:
                time.sleep(np.random.uniform(0, random_time))

            wandb.init(
                config=variant,
                project=project,
                dir=output_dir,
                id=experiment_id,
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                ),
            )


    def log(self, *args, **kwargs):
        if self.wandb_logging:
            wandb.log(*args, **kwargs)


class Timer(object):

    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            raise ValueError('Incorrect value type')
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def print_flags(flags, flags_def):
    logging.info(
        'Running training with hyperparameters: \n{}'.format(
            pprint.pformat(
                ['{}: {}'.format(key, getattr(flags, key)) for key in flags_def]
            )
        )
    )


def get_user_flags(flags, flags_def):
    return {key: getattr(flags, key) for key in flags_def}


def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }
