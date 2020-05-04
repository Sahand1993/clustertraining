# [(bs1, lr1), (bs2, lr2), (bs3, lr3)] (params)
# [1.17, 1.19, 2.20] (colours)

import os
import re
import pickle
from typing import List, Tuple
import matplotlib.pyplot as plt

regex = r"model_bs([0-9]+)_lr(.*)"

params: List[Tuple[int, float]] = []
maxAccs: List[float] = []
for dir in os.listdir("modelsRun1"):
    matchObj = re.match(regex, dir)
    params.append((float(matchObj.group(2)), int(matchObj.group(1))))

    # get the smallest validation loss achieved for the parameter pair
    losses: List[float] = pickle.load(open("./modelsRun1/" + dir + "/pickles/val_losses.pic", "rb"))
    accs: List[float] = pickle.load(open("./modelsRun1/" + dir + "/pickles/val_epoch_accs.pic", "rb"))
    maxAccs.append(max(accs))

print(params)
print(maxAccs)
maxAcc: float = max(maxAccs)
fig, ax = plt.subplots()
ax.scatter(x=list(map(lambda t: t[0], params)), y=list(map(lambda t: t[1], params)), c=list(map(lambda acc: acc/maxAcc, maxAccs)), cmap="YlOrRd")

for lr, bs in params:
    ax.annotate("{:.2E}".format(lr) + "," + str(bs), (lr, bs))

plt.show()