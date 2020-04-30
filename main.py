from batchiterators.fileiterators import *
from dssm.model_dense import *
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import sys
from helpers.helpers import correct_guesses_of_dssm
import random

saver = tf.compat.v1.train.Saver(max_to_keep=4)

def rnd(lower, higher):
  exp = random.randint(-higher, -lower)
  base = 0.9 * random.random() + 0.1
  return base * 10 ** exp

def get_feed_dict(batch):
  q_batch = batch.get_q_dense()
  p_batch = batch.get_relevant_dense()
  n1_batch, n2_batch, n3_batch, n4_batch = batch.get_irrelevant_dense()

  feed_dict = {
    x_q: q_batch,
    x_p: p_batch,
    x_n1: n1_batch,
    x_n2: n2_batch,
    x_n3: n3_batch,
    x_n4: n4_batch
  }

  return feed_dict

# learning rate and batch size
# Ranges:
#         lr: 0.01 to 0.000001
#         batch size: 16 to 2048 (in 2^n's, meaning 2^4 - 2^11)
try:
  learningRates = pickle.load(open("hyperparams/learningRates.pic", "rb"))
  batchSizes = pickle.load(open("hyperparams/batchSizes.pic", "rb"))
  SIZE = len(batchSizes)
  print("Loaded LRs and BSs")
except FileNotFoundError:
  print("Creating new LRs and BSs")
  SIZE = 8
  learningRates = [rnd(2, 5) for i in range(SIZE)]
  #learningRates = (np.random.random(SIZE)) * (0.01 - 0.000001) + 0.000001
  batchSizes = 2**np.random.randint(4, 11, SIZE)
  pickle.dump(learningRates, open("hyperparams/learningRates.pic", "wb"))
  pickle.dump(batchSizes, open("hyperparams/batchSizes.pic", "wb"))

print(learningRates)
print(batchSizes)

# Random Search for Hyperparameters
try:
  hyperItersDone = pickle.load(open("hyperparams/itersDone.pic", "rb"))
except FileNotFoundError:
  hyperItersDone = -1

#for paramPairNumber, (learningRate, batchSize) in enumerate(zip(learningRates, batchSizes)):
for i in range(hyperItersDone + 1, SIZE):
  learningRate, batchSize = learningRates[i], batchSizes[i]
  print("lr = {}, bs = {}".format(learningRate, batchSize))
  # Create hyperparameter-pair folder
  modelFolderPath = "models/{}".format("model_bs" + str(batchSize) + "_lr" + str(learningRate))
  
  try:
    os.mkdir(modelFolderPath)
  except FileExistsError:
    pass
  
  try:
    os.mkdir(modelFolderPath + "/pickles")
  except FileExistsError:
    pass

  try:
    os.mkdir(modelFolderPath + "/tf")
  except FileExistsError:
    pass

  with tf.compat.v1.Session() as sess:
    optimizer = tf.compat.v1.train.AdamOptimizer(learningRate).minimize(logloss)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    print("Initializing new models")    

    DENSE = True

    trainingSet = NaturalQuestionsFileIterator(
            "datasets/nq/train.csv",
            batch_size = batchSize,
            no_of_irrelevant_samples = 4,
            encodingType="NGRAM",
            dense=DENSE)
    validationSet = NaturalQuestionsFileIterator(
            "datasets/nq/validation.csv",
            batch_size=batchSize,
            no_of_irrelevant_samples=4,
            encodingType="NGRAM",
            dense=DENSE)
    
    EPOCHS = 30

    try:
      iterations_done = pickle.load(open(modelFolderPath + "/pickles/i.pic", "rb"))
      train_epoch_accuracies = pickle.load(open(modelFolderPath + "/pickles/train_epoch_accs.pic", "rb"))
      train_losses = pickle.load(open(modelFolderPath + "/pickles/train_losses.pic", "rb"))
      val_epoch_accuracies = pickle.load(open(modelFolderPath + "/pickles/val_epoch_accs.pic", "rb"))
      val_losses = pickle.load(open(modelFolderPath + "/pickles/val_losses.pic", "rb"))
      print("Got pickles:")
      print("iterations_done: {}".format(iterations_done))
      print("train_epoch_accuracies: {}".format(train_epoch_accuracies))
      print("val_epoch_accuracies: {}".format(val_epoch_accuracies))
      print("train_losses: {}".format(train_losses))
      print("val_losses: {}".format(val_losses))
      print("Starting on epoch " + str(iterations_done + 1))
    except FileNotFoundError as e:
      iterations_done = 0
      train_epoch_accuracies = []
      train_losses = []
      val_epoch_accuracies = []
      val_losses = []
      print("Couldn't find pickles: {}".format(e))
      print("Starting fresh")

    for epoch in range(iterations_done + 1, EPOCHS + 1):
      print("Epoch {}".format(epoch))
      if epoch > 0:
          trainingSet.restart()
          validationSet.restart()

      ll_train_overall = 0
      correct_train = 0
      for batch in tqdm(trainingSet):
          feed_dict = get_feed_dict(batch)
          _, ll = sess.run([optimizer, logloss], feed_dict=feed_dict)
          ll_train_overall += ll
          batch_correct = correct_guesses_of_dssm(sess, feed_dict, prob_p, prob_n1, prob_n2, prob_n3, prob_n4)
          correct_train += batch_correct

      #evaluate on validation set
      ll_val_overall = 0
      correct_val = 0
      for batch in validationSet:
          feed_dict = get_feed_dict(batch, DENSE)
          (ll_val,) = sess.run([logloss], feed_dict=feed_dict)
          batch_correct = correct_guesses_of_dssm(sess, feed_dict, prob_p, prob_n1, prob_n2, prob_n3, prob_n4)
          correct_val += batch_correct
          ll_val_overall += ll_val

      print(correct_train, trainingSet.getNoOfDataPoints())
      print(correct_val, validationSet.getNoOfDataPoints())
      val_losses.append(ll_val_overall / validationSet.getNoOfDataPoints())
      val_epoch_accuracies.append(correct_val / validationSet.getNoOfDataPoints())
      train_losses.append(ll_train_overall / trainingSet.getNoOfDataPoints())
      train_epoch_accuracies.append(correct_train / trainingSet.getNoOfDataPoints())

      saver.save(sess, modelFolderPath + "/tf/dssm", global_step=epoch)
      print("saved model")

      pickle.dump(train_losses, open(modelFolderPath + "/pickles/train_losses.pic", "wb"))
      pickle.dump(val_epoch_accuracies, open(modelFolderPath + "/pickles/val_epoch_accs.pic", "wb"))
      pickle.dump(val_losses, open(modelFolderPath + "/pickles/val_losses.pic", "wb"))
      pickle.dump(train_epoch_accuracies, open(modelFolderPath + "/pickles/train_epoch_accs.pic", "wb"))
      pickle.dump(epoch, open(modelFolderPath + "/pickles/i.pic", "wb"))
      print("Dumped losses and accuracies")

      print("overall train loss " + str(ll_train_overall / trainingSet.getNoOfDataPoints()))
      print("overall val loss " + str(ll_val_overall / validationSet.getNoOfDataPoints()))

    pickle.dump(i, open("hyperparams/itersDone.pic", "wb"))

    plt.figure()
    plt.plot(train_epoch_accuracies)
    plt.plot(val_epoch_accuracies)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("accuracies")
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel("epoch")
    plt.title("loss")
    plt.show()
    print("train accs: {}".format(train_epoch_accuracies))
    print("val accs: {}".format(val_epoch_accuracies))
    print("train losses: {}".format(train_losses))
    print("val losses: {}".format(val_losses))

# Save settings statistics in model folder for future comparison
