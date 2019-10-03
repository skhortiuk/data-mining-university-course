import logging
from copy import deepcopy
from random import random


class Kohonen:
    _logger = logging.getLogger()

    def __init__(self, neurons_count: int, inputs_count: int):
        self._neurons_count: int = neurons_count
        self._inputs_count: int = inputs_count
        self._network: list = [[random() for _ in range(inputs_count)] for _
                               in range(neurons_count)]

    @property
    def network(self) -> list:
        return self._network

    def is_trained(self, old_network: list,
                   max_neurons_diff: float = .005) -> bool:
        for new, old in zip(self._network, old_network):
            for new_elem, old_elem in zip(new, old):
                if abs(new_elem - old_elem) > max_neurons_diff:
                    return False

        return True

    def _get_winner_idx(self, data: list) -> int:
        min_distance: float = float("inf")
        winner: int = 0
        for i in range(self._neurons_count):
            distance: float = .0
            for j in range(self._inputs_count):
                distance += (self._network[i][j] - data[j]) ** 2

            if distance < min_distance:
                min_distance = distance
                winner = i

        return winner

    def train(self, dataset: list, train_rate: float = .6,
              change_factor: float = .95, max_diff: float = .005):
        self._logger.info(
            f"Start training model with train_rate ({train_rate}) and "
            f"change factor ({change_factor})"
        )
        trained: bool = False
        iterations: int = 0
        while not trained:
            self._logger.info(
                f"{iterations} iteration started"
            )
            old_network: list = deepcopy(self._network)
            for data in dataset:
                win_idx = self._get_winner_idx(data)
                for input_idx in range(self._inputs_count):
                    self._network[win_idx][input_idx] += train_rate * (
                            data[input_idx] - self._network[win_idx][input_idx]
                    )
            train_rate *= change_factor
            trained = self.is_trained(old_network, max_neurons_diff=max_diff)
            iterations += 1
        self._logger.info("Finished. Model ready!")

    def predict(self, value: list) -> int:
        return self._get_winner_idx(value)
