from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os

ITERS = 500

epsilons = {
    0.01: [],
    0.03: [],
    0.05: [],
    0.07: [],
    0.09: [],
    0.1: []
}

colors = ["r", "g", "b", "m", "c", "y"]

plt.ylabel("Scores")
plt.xlabel("Epochs")

for file in os.listdir("."):
    if not (os.path.isfile(file) and "hist-epsilon" in file and "npy" in file):
        continue
    for epsilon in epsilons.keys():
        if "%f" % epsilon in file:
            epsilons[epsilon].append(np.load(file))
            break

for epsilon in sorted(epsilons):
    print epsilon, len(epsilons[epsilon])


def compare_trials():
    for epsilon in sorted(epsilons):
        trials = epsilons[epsilon]
        percent_fail = []
        percent_fail_latter_half = []
        means = []
        # plt.boxplot(trials, whis=[5, 95])
        for trialNo, trial in enumerate(trials):
            mask = trial < 10
            pf = mask.sum() / ITERS
            pfh = mask[int(ITERS / 2):].sum() / (ITERS / 2)
            percent_fail.append(pf)
            percent_fail_latter_half.append(pfh)
            means.append(np.mean(trial))
            plt.scatter(
                range(ITERS),
                trial,
                color=colors[trialNo % len(colors)],
                label=str(trialNo) +
                ", %.2f%%, %.2f%%" % (pf * 100, pfh * 100))

        print epsilon, np.mean(means), np.mean(percent_fail), np.mean(percent_fail_latter_half)
        plt.title(
            "Visualizing variance for epsilon=%.2f trials" % epsilon)
        plt.legend(loc='upper left')
        # plt.show()


compare_trials()
exit(0)
