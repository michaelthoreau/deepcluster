import glob
import pandas as pd
import os
from matplotlib import pyplot as plt
fols = glob.glob("exp_k*")

for fol in fols:
	k = int(fol.split("k_")[-1].split("_")[0])
	try:
		df = pd.read_csv(os.path.join(fol, "log.csv"))
		# print(df)
	except:
		print("failed to load log")
		continue
	
	# fill first nmi with dummy data for plotting
	df['nmi_t'][0] = df['nmi_t'][1]
	

	# create figures directory
	figPath = os.path.join(fol, "figs")
	if not os.path.exists(figPath):
		os.makedirs(figPath)

	# plot nn loss
	plt.figure(0)
	plt.clf()
	plt.title("Network Loss (k = " + str(k) + ")")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.plot(df['epoch'], df['nn_loss'])
	plt.savefig(os.path.join(figPath, "nn_loss.png"))
	plt.savefig(os.path.join(figPath, "nn_loss.eps"))
	# plt.show()

	plt.figure(1)
	plt.clf()
	plt.title("Cluster Loss (k = " + str(k) + ")")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.plot(df['epoch'], df['cluster_loss'])
	plt.savefig(os.path.join(figPath, "cluster_loss.png"))
	plt.savefig(os.path.join(figPath, "cluster_loss.eps"))
	# plt.show()


	plt.figure(2)
	plt.clf()
	plt.title("Normalised Mutual Information t / t-1 (k = " + str(k) + ")")
	plt.xlabel("Epoch")
	plt.ylabel("NMI t/t-1")
	plt.plot(df['epoch'], df['nmi_t'])
	plt.savefig(os.path.join(figPath, "nmi_t.png"))
	plt.savefig(os.path.join(figPath, "nmi_t.eps"))
	# plt.show()

	plt.figure(3)
	plt.clf()
	plt.title("Normalised Mutual Information Between Clusters and Labels (k = " + str(k) + ")")
	plt.xlabel("Epoch")
	plt.ylabel("NMI Clusters / Labels")
	plt.plot(df['epoch'], df['nmi_labels'])
	plt.savefig(os.path.join(figPath, "nmi_labels.png"))
	plt.savefig(os.path.join(figPath, "nmi_labels.eps"))
	# plt.show()
