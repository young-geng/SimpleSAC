import os
import time
from copy import deepcopy

import numpy as np
import pprint

import gym
import torch

import absl.app
import absl.flags
from absl import logging

from .sac import SAC
from .replay_buffer import ReplayBuffer, batch_to_torch
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .utils import define_flags_with_default, set_random_seed, print_flags


flags_def = define_flags_with_default(
    env='HalfCheetah-v2',
    max_traj_length=1000,
    replay_buffer_size=100000,
    output_dir='/tmp/simple_sac',
    seed=42,
    device='cpu',
    logging_period=100,

    policy_arch='256-256',
    qf_arch='256-256',

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
    target_entropy=-3e-2,
    policy_lr=3e-4,
    qf_lr=3e-4,
    optimizer_type='adam',
    soft_target_update_rate=5e-3,
    target_update_period=1,
)


def main(argv):
    FLAGS = absl.flags.FLAGS
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    print_flags(FLAGS, flags_def)
    set_random_seed(FLAGS.seed)

    train_sampler = StepSampler(lambda: gym.make(FLAGS.env), FLAGS.max_traj_length)
    eval_sampler = TrajSampler(lambda: gym.make(FLAGS.env), FLAGS.max_traj_length)

    replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size)

    policy = TanhGaussianPolicy(
        train_sampler.env.observation_space.shape[0],
        train_sampler.env.action_space.shape[0],
        FLAGS.policy_arch
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

    sac = SAC(
        policy, qf1, qf2, target_qf1, target_qf2,
        discount=FLAGS.discount,
        reward_scale=FLAGS.reward_scale,
        alpha_multiplier=FLAGS.alpha_multiplier,
        use_automatic_entropy_tuning=FLAGS.use_automatic_entropy_tuning,
        target_entropy=FLAGS.target_entropy,
        policy_lr=FLAGS.policy_lr,
        qf_lr=FLAGS.qf_lr,
        optimizer_type=FLAGS.optimizer_type,
        soft_target_update_rate=FLAGS.soft_target_update_rate,
        target_update_period=FLAGS.target_update_period,
    )

    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    for epoch in range(FLAGS.n_epochs):
        start_time = time.time()

        train_sampler.sample(
            sampler_policy, FLAGS.n_env_steps_per_epoch,
            deterministic=False, replay_buffer=replay_buffer
        )

        sample_time = time.time() - start_time
        start_time = time.time()

        for batch in replay_buffer.generator(FLAGS.batch_size, FLAGS.n_train_step_per_epoch):
            batch = batch_to_torch(batch, FLAGS.device)
            sac.train(batch)

        train_time = time.time() - start_time
        start_time = time.time()

        if (epoch + 1) % FLAGS.eval_period == 0:
            trajs = eval_sampler.sample(
                sampler_policy, FLAGS.eval_n_trajs, deterministic=True
            )

            # TODO: add proper logging utilities.
            average_return = np.mean([np.sum(t['rewards']) for t in trajs])
            logging.info('Epoch: {}, average reward: {}'.format(epoch, average_return))

        eval_time = start_time - time.time()
        logging.info('Epoch: {}, sample time: {}, train time: {}, eval time: {}'.format(
            epoch, sample_time, train_time, eval_time
        ))


if __name__ == '__main__':
    absl.app.run(main)
