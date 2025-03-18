# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///

import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
y3 = relu(x)

plt.plot(x, y1, label="step function", linestyle="--")
plt.plot(x, y2, label="sigmoid", linestyle="-.")
plt.plot(x, y3, label="relu")
plt.ylim(-0.1, 1.1)
plt.show()
