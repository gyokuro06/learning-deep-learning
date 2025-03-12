# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
# ]
# ///

import numpy as np


def AND(x1: int, x2: int):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.7
    tmp = np.sum(w * x) - b
    if tmp >= 0:
        return 1
    else:
        return 0


print("AND")
print("x1 = 0, x2 = 0, y =", AND(0, 0))
print("x1 = 0, x2 = 1, y =", AND(0, 1))
print("x1 = 1, x2 = 0, y =", AND(1, 0))
print("x1 = 1, x2 = 1, y =", AND(1, 1))
print()


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = - 0.7
    tmp = np.sum(w * x) - b
    if tmp >= 0:
        return 1
    else:
        return 0


print("NAND")
print("x1 = 0, x2 = 0, y =", NAND(0, 0))
print("x1 = 0, x2 = 1, y =", NAND(0, 1))
print("x1 = 1, x2 = 0, y =", NAND(1, 0))
print("x1 = 1, x2 = 1, y =", NAND(1, 1))
print()


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1.0, 1.0])
    b = 1
    tmp = np.sum(w * x) - b
    if tmp >= 0:
        return 1
    else:
        return 0


print("OR")
print("x1 = 0, x2 = 0, y =", OR(0, 0))
print("x1 = 0, x2 = 1, y =", OR(0, 1))
print("x1 = 1, x2 = 0, y =", OR(1, 0))
print("x1 = 1, x2 = 1, y =", OR(1, 1))
print()


def XOR(x1, x2):
    a = NAND(x1, x2)
    b = OR(x1, x2)
    return AND(a, b)


print("XOR")
print("x1 = 0, x2 = 0, y =", XOR(0, 0))
print("x1 = 0, x2 = 1, y =", XOR(0, 1))
print("x1 = 1, x2 = 0, y =", XOR(1, 0))
print("x1 = 1, x2 = 1, y =", XOR(1, 1))
print()
