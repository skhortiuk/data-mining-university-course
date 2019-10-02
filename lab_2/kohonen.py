from random import random


class Kohonen:
    def __init__(self, neurons_count: int, num_classes: int):
        self._neurons: list = []
        for i_idx in range(neurons_count):
            self._neurons.append([random() for _ in range(num_classes)])

    def is_trained(self, old_neurons: list,
                   max_neurons_diff: float = .005) -> bool:
        for new, old in zip(self._neurons, old_neurons):
            for new_elem, old_elem in zip(new, old):
                if abs(new_elem - old_elem) > max_neurons_diff:
                    return False

        return True

    def train(self, dataset: list):
        old_neurons: list = []
        trained_value: float = 0.6
