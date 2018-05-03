from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os

ITERS = 500

discounts = {
    0.1: [],
    0.4: [],
    0.5: [],
    0.6: [],
    0.7: [],
    0.8: [],
    0.9: []
}

colors = ["r", "g", "b", "m", "c", "y"]

plt.ylabel("Scores")
plt.xlabel("Epochs")

for file in os.listdir("."):
    if not (os.path.isfile(file) and "hist-discount" in file and "npy" in file):
        continue
    for discount in discounts.keys():
        if "%f" % discount in file:
            discounts[discount].append(np.load(file))
            break

for discount in sorted(discounts):
    print discount, len(discounts[discount])


def compare_trials():
    for discount in sorted(discounts):
        trials = discounts[discount]
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

        print discount, np.mean(means), np.mean(percent_fail), np.mean(percent_fail_latter_half)
        # plt.title(
        #     "Visualizing variance for discount=%.2f trials" % discount)
        # plt.legend(loc='upper left')
        # plt.savefig("variances/discount/bw-%.2f.png" % discount)


compare_trials()
