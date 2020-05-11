import glob
import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
fols = glob.glob("exp_k*/supervised_*")

precision_k = []
precision_split = []
precision_split_rand = []
for fol in fols:
    k = int(fol.split("k_")[-1].split("_")[0])
    train_split = float(fol.split("_")[-1])
    # print(fol, k, train_split)
    try:
    	df = pd.read_csv(os.path.join(fol, "log_supervised_.csv"))
    	# print(df)
    except:
    	print("failed to load log")
    	continue
    
    print("k: {:5d} train_split: {:1.5f} best precision: {:1.4f}".format(k, train_split, np.max(df['prec1'].to_numpy())))
    if train_split == 1.0 and not "rand" in fol:
        precision_k.append([k, np.max(df['prec1'].to_numpy())])
    

    if k == 1000 and train_split >= 0.001:
        if "rand" in fol:
            precision_split_rand.append([train_split, np.max(df['prec1'].to_numpy())])
        else:
            precision_split.append([train_split, np.max(df['prec1'].to_numpy())])

    # create figures directory
    figPath = os.path.join(fol, "figs")
    if not os.path.exists(figPath):
    	os.makedirs(figPath)

    # # plot nn loss
    # plt.figure(0)
    # plt.clf()
    # plt.title("Network Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.plot(df['epoch'], df['loss'])
    # plt.savefig(os.path.join(figPath, "loss.png"))
    # plt.savefig(os.path.join(figPath, "loss.eps"))
    # # plt.show()

    # plt.figure(1)
    # plt.clf()
    # plt.title("Top-1 Precision")
    # plt.xlabel("Epoch")
    # plt.ylabel("top-1 precision")
    # plt.plot(df['epoch'], df['prec1'])
    # plt.savefig(os.path.join(figPath, "prec1.png"))
    # plt.savefig(os.path.join(figPath, "prec1.eps"))
    # # plt.show()


    # plt.figure(2)
    # plt.clf()
    # plt.title("Top-5 Precision")
    # plt.xlabel("Epoch")
    # plt.ylabel("top-5 precision")
    # plt.plot(df['epoch'], df['prec5'])
    # plt.savefig(os.path.join(figPath, "prec5.png"))
    # plt.savefig(os.path.join(figPath, "prec5.eps"))
    # # plt.show()






precision_k = sorted(precision_k, key=lambda x: (x[0],x[1]))

plt.figure(7)
ax = plt.gca()
plt.plot(np.array(precision_k)[:,0], np.array(precision_k)[:,1], "x--")
plt.title("Effect of k on supervised precision")
plt.xlabel("k")
plt.ylabel("Precision")
ax.set_xscale('log')
# plt.show()


precision_split = sorted(precision_split, key=lambda x: (x[0],x[1]))
precision_split_rand = sorted(precision_split_rand, key=lambda x: (x[0],x[1]))
print(precision_split)
print(precision_split_rand)
plt.figure(8)
ax = plt.gca()
plt.plot(np.array(precision_split)[:,0], np.array(precision_split)[:,1], "x--")
plt.plot(np.array(precision_split_rand)[:,0], np.array(precision_split_rand)[:,1], "o--r")
plt.title("Effect of available training data on supervised precision")
plt.xlabel("Supervised data available (X 60000)")
plt.ylabel("Precision")
plt.legend(['Unsupervised Pre-Training', 'Random Initialisation'])
ax.set_xscale('log')
# ax.ticklabel_format(style='plain')
# ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
# ax.ticklabel_format(useOffset=False, style='plain')

plt.show()
