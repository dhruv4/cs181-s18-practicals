import numpy as np
import matplotlib.pyplot as plt

ITERS = 500

hist5 = np.load("hist-discount-0.500000.npy")
hist6 = np.load("hist-discount-0.600000.npy")
hist7 = np.load("hist-discount-0.700000.npy")
hist9 = np.load("hist-discount-0.900000.npy")
hist10 = np.load("hist-discount-1.000000.npy")

plt.ylabel("Scores")
plt.xlabel("Epochs")

a = [
    # (hist1, "r", "0.01"),
    # (hist3, "g", "0.03"),
    # (hist5, "b", "0.05"),
    # (hist7, "m", "0.07"),
    (hist5, "m", "0.5"),
    (hist6, "c", "0.6"),
    (hist7, "y", "0.7"),
    (hist9, "g", "0.9"),
    (hist10, "r", "1.0")
]

for h in a:
    plt.scatter(range(len(h[0])), h[0], color=h[1], label=h[2])

plt.title("Testing Various Discount Values, epsilon=0.09")
plt.legend(loc='upper left')
plt.show()
