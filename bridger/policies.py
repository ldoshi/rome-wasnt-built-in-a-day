import abc
import numpy as np


class Policy(abc.ABC):
    """Policy base class"""

    def __call__(self, state, *args, **kwargs):
        result = self.run(state, *args, **kwargs)
        assert result is not None
        return result

    @abc.abstractmethod
    def run(self, state, *args, **kwargs):
        pass


class SamplingPolicy(Policy):
    """Policy that returns an action by sampling from a distribution"""

    def run(self, state, *args, **kwargs):
        probabilities = self.get_probabilities(state, *args, **kwargs)
        return np.random.choice(probabilities.shape[0], p=probabilities)

    @abc.abstractmethod
    def get_probabilities(self, state, *args, **kwargs):
        pass


class EstimatorPolicy(SamplingPolicy):
    """Policy that uses a Q-function approximator"""

    def __init__(self, estimator):
        """Stores the `estimator` callable in the instance variable `Q`

        Args:
            estimator: A callable that takes a state as input and returns a
            numpy array of per-action Q values for the input state
        """
        assert callable(estimator)
        self.Q = estimator


class EpsilonGreedyPolicy(EstimatorPolicy):
    """Policy that, for a given state S, selects an action as follows:

    With probability `1-epsilon`, this policy will select A, the action with
    maximal Q-value for state S. Otherwise, the policy will select an action
    uniformly at random from the action space"""

    def __init__(self, estimator, epsilon=0):
        """See docstrings from superclass and class docstring for descriptions
        of `estimator` and `epsilon` respectively"""

        super().__init__(estimator)
        self.epsilon = epsilon

    def get_probabilities(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        q_values = self.Q(state).squeeze()
        num_actions = q_values.shape[0]
        probabilities = np.full(num_actions, epsilon / num_actions)
        probabilities[q_values.argmax()] += 1 - epsilon
        return probabilities


class GreedyPolicy(EpsilonGreedyPolicy):
    """Policy that, for a given state S, selects the action with best Q-value"""

    def __init__(self, estimator):
        """See docstring from superclass EstimatorPolicy for description of `estimator`"""
        super().__init__(estimator)

    def get_probabilities(self, state):
        return super().get_probabilities(state)
