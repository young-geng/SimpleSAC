import os
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import gym
import torch

import absl.app
import absl.flags

from .sac import SAC
from .replay_buffer import ReplayBuffer
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .utils import Timer, define_flags_with_default, set_random_seed
from .utils import WandBLogger, print_flags, get_user_flags, prefix_metrics
from viskit.logging import logger, setup_logger


FLAGS_DEF = define_flags_with_default(
    env='HalfCheetah-v2',
    max_traj_length=1000,
    replay_buffer_size=1000000,
    output_dir='/tmp/simple_sac',
    seed=42,
    device='cpu',

    policy_arch='256-256',
    qf_arch='256-256',
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=2000,
    n_env_steps_per_epoch=1000,
    n_train_step_per_epoch=1000,
    eval_period=20,
    eval_n_trajs=5,

    batch_size=256,

    discount=0.99,
    reward_scale=1.0,
    alpha_multiplier=1.0,
    use_automatic_entropy_tuning=True,
    target_entropy=0.0,
    policy_lr=3e-4,
    qf_lr=3e-4,
    optimizer_type='adam',
    soft_target_update_rate=5e-3,
    target_update_period=1,

    wandb_logging=False,
    wandb_prefix='SimpleSAC',
    wandb_project='sac',
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    experiment_id = uuid.uuid4().hex
    setup_logger(
        variant=variant,
        exp_id=experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.output_dir,
        include_exp_prefix_sub_dir=False
    )

    wandb_logger = WandBLogger(
        wandb_logging=FLAGS.wandb_logging,
        variant=variant,
        project=FLAGS.wandb_project,
        experiment_id=experiment_id,
        prefix=FLAGS.wandb_prefix,
        output_dir=logger.get_snapshot_dir(),
    )

    set_random_seed(FLAGS.seed)

    train_sampler = StepSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)

    replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size)

    policy = TanhGaussianPolicy(
        train_sampler.env.observation_space.shape[0],
        train_sampler.env.action_space.shape[0],
        FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
    )

    qf1 = FullyConnectedQFunction(
        train_sampler.env.observation_space.shape[0],
        train_sampler.env.action_space.shape[0],
        FLAGS.qf_arch
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        train_sampler.env.observation_space.shape[0],
        train_sampler.env.action_space.shape[0],
        FLAGS.qf_arch
    )
    target_qf2 = deepcopy(qf2)

    if FLAGS.target_entropy >= 0:
        target_entropy = -np.prod(train_sampler.env.action_space.shape).item()
    else:
        target_entropy = FLAGS.target_entropy

    sac = SAC(
        policy, qf1, qf2, target_qf1, target_qf2,
        discount=FLAGS.discount,
        reward_scale=FLAGS.reward_scale,
        alpha_multiplier=FLAGS.alpha_multiplier,
        use_automatic_entropy_tuning=FLAGS.use_automatic_entropy_tuning,
        target_entropy=target_entropy,
        policy_lr=FLAGS.policy_lr,
        qf_lr=FLAGS.qf_lr,
        optimizer_type=FLAGS.optimizer_type,
        soft_target_update_rate=FLAGS.soft_target_update_rate,
        target_update_period=FLAGS.target_update_period,
    )
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    metrics = {}
    for epoch in range(FLAGS.n_epochs):

        with Timer() as rollout_timer:
            train_sampler.sample(
                sampler_policy, FLAGS.n_env_steps_per_epoch,
                deterministic=False, replay_buffer=replay_buffer
            )
            metrics['env_steps'] = replay_buffer.total_steps
            metrics['epoch'] = epoch

        with Timer() as train_timer:
            batch_generator = replay_buffer.torch_sample_generator(
                FLAGS.batch_size, FLAGS.device, FLAGS.n_train_step_per_epoch
            )
            for batch in batch_generator:
                metrics.update(prefix_metrics(sac.train(batch, return_stats=True), 'sac'))

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])

        metrics['rollout_time'] = rollout_timer()
        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        wandb_logger.log(metrics)


if __name__ == '__main__':
    absl.app.run(main)
