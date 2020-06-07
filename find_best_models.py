import os
import pickle
import sys

basedir = sys.argv[1]

for _dir in os.listdir(basedir):
    model_home_dir = list(filter(lambda dir_name: not dir_name.endswith("checkpoints"), os.listdir(os.path.join(basedir, _dir))))[0]
    print(_dir)
    try:
        val_accs = pickle.load(open(os.path.join(basedir, _dir, model_home_dir, "pickles", "val_epoch_accs.pic"), "rb"))
    except Exception as e:
        print(e)
        continue
    if len(val_accs) < 20:
        print("NOT FINISHED, run more epochs")
        continue
    print(val_accs)
    keep_model_no = val_accs.index(max(val_accs)) + 1
    print(os.path.join(basedir, _dir, model_home_dir))
    print("keep model no {} which has max val acc {}".format(keep_model_no, max(val_accs)))
    print()