from queue import Queue
import threading

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, max_size):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self, observation, action, reward, next_observation, done):
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(next_observation, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.add(o, a, r, no, d)

    def sample(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=batch_size > len(self))
        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...]
        )

    def generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self):
        return self._total_steps


def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }


class CachedIterator(object):

    def __init__(self, iterator, map_fn=None, cache_size=10):
        self.iterator = iterator
        self.map_fn = map_fn
        self.cache_size = cache_size

        self.queue = None
        self.worker = None

    def _enqueue(self, queue, iterator, map_fn):
        for obj in iterator:
            if map_fn is not None:
                obj = map_fn(obj)
            queue.put((False, obj))
        queue.put((True, None))

    def __iter__(self):
        if self.cache_size == 0:
            for obj in self.iterator:
                if self.map_fn is not None:
                    obj = self.map_fn(obj)
                yield obj
        else:
            if self.queue is None:
                self.queue = Queue(self.cache_size)
                self.worker = threading.Thread(
                    target=self._enqueue,
                    args=(self.queue, self.iterator, self.map_fn)
                )
                self.worker.start()
            while True:
                task_done, obj = self.queue.get()
                if task_done:
                    self.worker.join()
                    raise StopIteration
                else:
                    yield obj
