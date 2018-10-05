'''
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht
'''

import parser
import time
import os
import re
import numpy as np
import gym
import logz
import ray
import utils
import optimizers
from policies import *
import socket
from shared_noise import *
from osim.env import ProstheticsEnv

@ray.remote
class Worker(object):
    """
    Object class for parallel rollout generation.
    """

    def __init__(self,
                 env_seed,
                 env_name='',
                 policy_params=None,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.02,
                 param_restore=None):

        # initialize OpenAI environment for each worker
        self.env = ProstheticsEnv(False, 1e-3, seed=env_seed)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table.
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params, param_restore)
        elif policy_params['type'] == 'MLP':
            self.policy = MLPPolicy(policy_params, param_restore)
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError

        self.delta_std = delta_std
        self.rollout_length = rollout_length


    def get_weights_plus_stats(self):
        """
        Get current policy weights and current statistics of past states.
        """
        return self.policy.get_weights_plus_stats()


    def rollout(self, shift = 0., rollout_length = None):
        """
        Performs one rollout of maximum length rollout_length.
        At each time-step it substracts shift from the reward.
        """

        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0
        obs = []
        actions = []

        ob = self.env.reset()
        for i in range(rollout_length):
            action = self.policy.act(ob)
            ob, reward, done, _ = self.env.step(action)

            obs.append(ob)
            actions.append(action)

            steps += 1
            total_reward += (reward - shift)
            if done:
                break

        return total_reward, steps, obs, actions

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        """
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, rollout_weights,
            rollout_filters, deltas_idx = [], [], [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)

                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps, obs, actions = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
                rollout_rewards.append(reward)

            else:
                idx, delta = self.deltas.get_delta(w_policy.size)

                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_filter = self.policy.get_observation_filter()
                pos_reward, pos_steps, pos_obs, pos_actions  = self.rollout(shift = shift)

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                neg_filter = self.policy.get_observation_filter()
                neg_reward, neg_steps, neg_obs, neg_actions = self.rollout(shift = shift)
                steps += pos_steps + neg_steps

                rollout_rewards.append([pos_reward, neg_reward])
                rollout_filters.append([pos_filter, neg_filter])
                rollout_weights.append([w_policy + delta, w_policy - delta])

        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards,
                "steps" : steps, 'rollout_filters': rollout_filters,
                'rollout_weights': rollout_weights}

    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()

    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    def get_delta(self):
        return self.deltas.get_delta(self.policy_params['ob_dim']*self.policy_params['ac_dim'])


class ARSLearner(object):
    """
    Object class implementing the ARS algorithm.
    """

    def __init__(self, env_name='HalfCheetah-v1',
                 policy_params=None,
                 num_workers=32,
                 num_deltas=320,
                 deltas_used=320,
                 delta_std=0.02,
                 logdir=None,
                 rollout_length=1000,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 seed=123,
                 restore=None):

        logz.configure_output_dir(logdir)
        logz.save_params(params)

        env = ProstheticsEnv(False, 1e-3)
        if restore is not None:
            param_restore = self.restore(restore)
            self.curr_iter = int(re.search('\d+', restore).group())
        else:
            param_restore = None
            self.curr_iter = 0

        self.max_reward = 0
        self.timesteps = 0
        self.action_size = env.action_space.shape[0]
        self.ob_size = env.observation_space.shape[0]
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')


        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = seed + 3)
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.')
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      env_name=env_name,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std,
                                      param_restore=param_restore) for i in range(num_workers)]


        # initialize policy
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params, param_restore)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'MLP':
            self.policy = MLPPolicy(policy_params, param_restore)
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError

        # adjust shared table
        if restore is not None:
            remove_idx = []
            for _ in range(self.curr_iter):
                for worker in self.workers:
                    idx, _ = ray.get(worker.get_delta.remote())
                    remove_idx.append(idx)
            print("REMOVED IDXS", remove_idx)

        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)
        print("Initialization of ARS complete.")

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts

        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)

        # parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = num_rollouts,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]

        # gather results
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx = [], []

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        max_idx = np.argmax(rollout_rewards)
        print('Maximum reward of collected rollouts:', rollout_rewards.max())

        logz.log_tabular("Maximum reward", rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)
        logz.log_tabular("Time taken", t2 - t1)

        if evaluate:
            return rollout_rewards

        logz.log_tabular("timesteps", self.timesteps)
        logz.dump_tabular()

        # select top performing directions if deltas_used < num_deltas
        max_rewards = rollout_rewards[max_idx]
        if max_rewards > self.max_reward:
            self.max_reward = max_rewards
            max_w = rollout_weights[max_idx]
            max_f = rollout_filters[max_idx]

            np.save(self.logdir + f'/max_params_{self.curr_iter}',
                    [w, f], allow_pickle=True)

        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas


        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx,:]

        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size = 500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat

    def save(self, i):
        """
        Save weights, optimizer parameters.
        """
        w = self.policy.get_weights()
        f = self.policy.observation_filter

        np.save(self.logdir + f'/params_{i}', [w, f],
                allow_pickle=True)
        return

    def restore(self, file_path):
        return np.load(file_path)

    def train_step(self):
        """
        Perform one update step of the policy weights.
        """

        g_hat = self.aggregate_rollouts()
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        return

    def train(self, num_iter):

        start = time.time()
        for i in range(self.curr_iter, self.curr_iter+num_iter):
            # make sure master filter buffer is clear
            self.policy.observation_filter.clear_buffer()

            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)

            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)

            t1 = time.time()
            self.train_step()
            t2 = time.time()

            print('total time of one step', t2 - t1)
            print('iter ', i,' done')

            # get statistics from all workers
            for j in range(self.num_workers):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy.observation_filter.stats_increment()

            # record statistics every 10 iterations
            if ((i + 1) % 1 == 0):
                self.save(i+1)

        return

def run_ars(params):

    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    env = ProstheticsEnv(False, 1e-3)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type':'linear',
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim}

    ARS = ARSLearner(env_name=params['env_name'],
                     policy_params=policy_params,
                     num_workers=params['n_workers'],
                     num_deltas=params['n_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'],
                     logdir=logdir,
                     rollout_length=params['rollout_length'],
                     shift=params['shift'],
                     params=params,
                     seed=params['seed'],
                     restore=params['restore'])

    ARS.train(params['n_iter'])

    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    parser.add_argument('--n_iter', '-n', type=int, default=1000000)
    parser.add_argument('--n_directions', '-nd', type=int, default=71)
    parser.add_argument('--deltas_used', '-du', type=int, default=71)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=0.075)
    parser.add_argument('--n_workers', '-e', type=int, default=71)
    parser.add_argument('--rollout_length', '-r', type=int, default=300)

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str,
                        default='/home/medipixel/ARS')

    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')
    parser.add_argument('--restore', type=str, default=None)

    ray.init()

    args = parser.parse_args()
    params = vars(args)
    run_ars(params)
