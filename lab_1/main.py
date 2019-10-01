from lab_1.perceptron import Perceptron


def get_correct_value(data: list) -> int:
    return int(not (data[0] or data[1]) or data[2])


dataset: list = [
    [1, 1, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 0, 0],
]

perceptron: Perceptron = Perceptron(3, threshold=.5, learning_rate=.4)
iteration: int = 0

expected = [get_correct_value(d) for d in dataset]
actual = []
while actual != expected and iteration < 1000:
    actual = []
    for data in dataset:
        correct_vale = get_correct_value(data)
        result: bool = perceptron.train(data, correct_vale)
        actual.append(int(result))
        iteration += 1
    print(f"\r{expected} - Expected \n\r\r{actual} - Actual", end=' ')
