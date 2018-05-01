import numpy as np
import matplotlib.pyplot as plt
import os

ITERS = 500

epsilons = {0.01: [],
            0.03: [],
            0.05: [],
            0.07: [],
            0.09: [],
            0.1: []}

colors = ["r", "g", "b", "m", "c", "y"]

plt.ylabel("Scores")
plt.xlabel("Epochs")

for epsilon in epsilons.keys():
    print epsilon
    files = [f for f in os.listdir(".") if os.path.isfile(f) and
             "%f" % epsilon in f and
             "hist-epsilon" in f and
             ".npy" in f]
    print files
    for f in files:
        epsilons[epsilon].append(np.load(f))


def compare_trials():
    for epsilon, trials in epsilons.iteritems():
        for trialNo, trial in enumerate(trials):
            plt.scatter(
                range(ITERS),
                trial,
                color=colors[trialNo],
                label=str(trialNo))

        plt.title(
            "Visualizing variance for epsilon=%.2f trials" % epsilon)
        plt.legend(loc='upper left')
        plt.show()


exit(0)

hist5 = np.load("hist-epsilon-0.050000.npy")
hist50 = np.load("hist-epsilon-0.050000-10.npy")
hist51 = np.load("hist-epsilon-0.050000-11.npy")
hist3 = np.load("hist-epsilon-0.030000.npy")
hist7 = np.load("hist-epsilon-0.070000.npy")
hist8 = np.load("hist-epsilon-0.080000.npy")
hist9 = np.load("hist-epsilon-0.090000.npy")
hist1 = np.load("hist-epsilon-0.010000.npy")
hist10 = np.load("hist.npy")

a = [
    # (hist1, "r", "0.01"),
    # (hist3, "g", "0.03"),
    (hist5, "b", "0.05"),
    (hist50, "c", "new"),
    (hist51, "r", "new2"),
    # (hist7, "m", "0.07"),
    # (hist8, "m", "0.08"),
    # (hist9, "c", "0.09"),
    # (hist10[:500], "y", "0.1")
]

for h in a:
    plt.scatter(range(len(h[0])), h[0], color=h[1], label=h[2])

plt.title("Narrowing Down Optimal Epsilon Value, discount=0.9")
plt.legend(loc='upper left')
plt.show()
