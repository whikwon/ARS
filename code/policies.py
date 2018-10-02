'''
Policy class for computing action from weights and observation vector.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht
'''


import numpy as np
from filter import get_filter

class Policy(object):

    def __init__(self, policy_params, param_restore):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True

    def update_weights(self, new_weights):
        self.weights = new_weights
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params, param_restore):
        Policy.__init__(self, policy_params, param_restore)
        if param_restore is not None:
            self.weights = param_restore[0]
            print("Pre-trained weights Restored")
            self.observation_filter.sync(param_restore[1])
            print("Observation filter Restored")
        else:
            self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype=np.float32)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux


class MLPPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params, param_restore):
        Policy.__init__(self, policy_params, param_restore)
        if param_restore is not None:
            self.weights = param_restore[0]
            print("Pre-trained weights Restored")
            self.observation_filter.sync(param_restore[1])
            print("Observation filter Restored")
        else:
            self.weights = np.zeros((512, self.ac_dim+self.ob_dim), dtype=np.float32)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        weight1 = self.weights[:, :self.ob_dim]
        weight2 = self.weights[:, self.ob_dim:]
        return np.dot(np.dot(weight1, ob), weight2)
