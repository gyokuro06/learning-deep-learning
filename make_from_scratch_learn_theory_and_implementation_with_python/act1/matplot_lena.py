# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///


import matplotlib.pyplot as plt
from matplotlib.image import imread


img = imread("lena.png")
plt.imshow(img)

plt.show()
