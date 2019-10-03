import logging

from lab_2.kohonen import Kohonen

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

dataset = [
    # Volodimirovich
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    # Khortiuk
    [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    # Serhii
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
]

kohonen = Kohonen(3, 14)
kohonen.train(dataset, train_rate=.5, change_factor=.6, max_diff=5e-4)
print("[*] Trained model:")
print(*kohonen.network, sep="\n")
print("[*] Predictions:")
for data in dataset:
    print(f"{data} is {kohonen.predict(data)} class")
