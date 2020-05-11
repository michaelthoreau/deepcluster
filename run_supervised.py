import glob
import subprocess
import time
unsupList = glob.glob("*/checkpoints/checkpoint_190.pth.tar")
trainSetList = glob.glob("mnist_split/*")

processList = []
for u in unsupList:
    for t in trainSetList:
        while len(processList) >= 5:
            for proc in processList:
                if proc.poll() != None:
                    processList.remove(proc)
            time.sleep(1.0)
        
        command = "python deepcluster/eval_linear.py --data_train " + t + " --data_val mnist_png/testing/ --model " + u + " --verbose"
        proc=subprocess.Popen(command.split(" "))
        processList.append(proc)
        # ls_output.communicate()  # Will block for 30 seconds

while len(processList) >= 5:
            for proc in processList:
                if proc.poll() != None:
                    processList.remove(proc)
            time.sleep(1.0)
print("DONE")

# python deepcluster/eval_linear.py --data_train mnist_split/training_0.0001/ --data_val mnist_png/testing/ --model exp_k_1000_lr_0.05_epochs_200/checkpoints/checkpoint_190.pth.tar --verbose
