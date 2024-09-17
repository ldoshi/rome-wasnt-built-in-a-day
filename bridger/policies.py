import abc
import torch
from typing import Optional
import numpy as np
import math


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
        # TODO: This is fine while we plan on doing Q-Learning, but if we ever
        # want to differentiate through our policies, we'll have to replace
        # this torch.multinomial# call with a torch module that uses the
        # reparameterization trick
        return torch.multinomial(probabilities, 1).item()

    @abc.abstractmethod
    def get_probabilities(self, state, *args, **kwargs):
        pass


class EstimatorPolicy(SamplingPolicy):
    """Policy that uses a Q-function approximator"""

    def __init__(self, estimator):
        """Stores the `estimator` callable in the instance variable `Q`

        Args:
            estimator: A callable that takes a state as input and returns a
            torch tensor of per-action Q values for the input state
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
        exploit = torch.where(q_values == q_values.max(), 1, 0)
        explore = torch.full(q_values.shape[0:], 1 / q_values.shape[0])
        probabilities = (1 - epsilon) * exploit + epsilon * explore
        return probabilities


def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    def __init__(self, prior=0):
        self._shape: Optional[tuple[int, int]] = None
        self._state: Optional[tuple[list]] = None
        self._value: int = 0
        self._visit_count: int = 0
        self._children = {}
        self._prior = prior

    def set_shape(self, shape: tuple[int, int]) -> None:
        self._shape = shape

    def get_shape(self) -> tuple[int, int]:
        return self._shape

    def set_state(self, state: tuple[list[int]]):
        self._state = state

    def get_shape(self) -> tuple[list[int]]:
        return self._state

    def set_visit_count(self, visit_count: int) -> None:
        self._visit_count = visit_count

    def get_visit_count(self) -> int:
        return self._visit_count

    def is_leaf(self):
        return len(self.children) == 0

    def value(self):
        if self._visit_count == 0:
            return 0
        return self._value / self._visit_count

    def select_action(self, temperature):
        """
        Selects an action based on the visit count distribution and temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self._children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob)


class MonteCarloTreeSearchPolicy(EstimatorPolicy):
    """Policy that uses a Monte Carlo Tree to sample from a distribution."""

    def __init__(self, estimator, environment):
        super().__init__(estimator)
        # Initialize the tree with an empty root node.
        self._root = Node()
        self._cur = self._root

    def calculate_simulation(self, model, state, num_simulations):
        """
        Update the tree state and visit counts for a given simulation.
        """
        action_probs, value = model.predict(state)
        self._root.expand(state, action_probs)

        for _ in range(num_simulations):
            node = self._root
            search_path = [node]

            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.get_state()

            next_state, reward, done, _ = self.env.step(action)
            if reward == -1:
                action_probs, value = model.predict(next_state)
                node.expand(next_state, action_probs)

            self.backpropagate(search_path, value)

        return self._root

    def backpropagate(self, search_path, value):
        """
        After a simulation, we propagate the evaluation back to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1

    def get_probabilities(self, state, epsilon, temperature: int) -> torch.tensor:
        if epsilon is None:
            epsilon = self.epsilon
        q_values = self.Q(state).squeeze()
        exploit = torch.where(q_values == q_values.max(), 1, 0)
        explore = torch.full(q_values.shape[0:], 1 / q_values.shape[0])
        probabilities = (1 - epsilon) * exploit + epsilon * explore
        return probabilities


class GreedyPolicy(EpsilonGreedyPolicy):
    """Policy that, for a given state S, selects the action with best Q-value"""

    def __init__(self, estimator):
        """See docstring from superclass EstimatorPolicy for description of `estimator`"""
        super().__init__(estimator)

    def get_probabilities(self, state):
        return super().get_probabilities(state)


choices = {
    "greedy": GreedyPolicy,
    "epsilon_greedy": EpsilonGreedyPolicy,
    "monte_carlo_tree_search": MonteCarloTreeSearchPolicy,
}
