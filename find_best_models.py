import os
import pickle
import sys
from typing import List, Tuple

basedir = sys.argv[1]

def find_index_with_maxacc_and_minloss(val_accs, val_losses):
    maxAcc = max(val_accs)
    indicesWithMaxAcc: List[Tuple[int, float]] = []
    for index in range(len(val_accs)):
        if (val_accs[index] == maxAcc):
            indicesWithMaxAcc.append((index, val_losses[index]))

    return min(indicesWithMaxAcc, key=lambda tuple: tuple[1])[0]

for _dir in os.listdir(basedir):
    model_home_dir = list(filter(lambda dir_name: not dir_name.endswith("checkpoints"), os.listdir(os.path.join(basedir, _dir))))[0]
    print(_dir)
    try:
        val_accs = pickle.load(open(os.path.join(basedir, _dir, model_home_dir, "pickles", "val_epoch_accs.pic"), "rb"))
        val_losses = pickle.load(open(os.path.join(basedir, _dir, model_home_dir, "pickles", "val_losses.pic"), "rb"))
    except Exception as e:
        print(e)
        continue
    if len(val_accs) < 20:
        print("NOT FINISHED, run more epochs")
        continue
    print(val_accs)
    print(val_losses)
    keep_model_no = find_index_with_maxacc_and_minloss(val_accs, val_losses) + 1

    print("model {} has greatest accuracy and smallest loss for that accuracy".format(keep_model_no))

    print(os.path.join(basedir, _dir, model_home_dir))
    #print("keep model no {} which has max val acc {}".format(keep_model_no, max(val_accs)))
    print()