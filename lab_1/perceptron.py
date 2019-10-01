import random


class Perceptron:
    def __init__(self, neurons_count: int, threshold: float = .5,
                 learning_rate: float = .2):
        self._threshold: float = threshold
        self._learning_rate: float = learning_rate
        self._neurons: list = [random.random() for _ in range(neurons_count)]

    def train(self, data: list, expected: int) -> bool:
        if len(self._neurons) != len(data):
            raise ValueError(
                f"Expected length of `data` is {len(self._neurons)}."
            )

        acc: float = .0
        for neuron, d in zip(self._neurons, data):
            acc += neuron * d

        actual: int = int(acc > self._threshold)

        if actual != expected:
            difference: float = expected - self._threshold
            for idx in range(len(self._neurons)):
                delta = self._learning_rate * difference * data[idx]
                self._neurons[idx] += delta

        return actual
