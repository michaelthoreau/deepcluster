# copy data for supervised training
import glob
import os
from random import sample
import shutil
import numpy as np
# minisets = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
minisets = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


for split in minisets:
  fol = "mnist_split/training_" + str(split)
  if not os.path.exists(fol):
    os.makedirs(fol)
  else:
    shutil.rmtree(fol)
    os.makedirs(fol)
  print(split)
  for d in digits:
    files = glob.glob("mnist_png/training/" + str(d) + "/*.png")
    imgs = sample(files, int(np.ceil(split*len(files))))

    for img in imgs:
      imageFol = os.path.join(fol, os.path.dirname(img).split('/')[-1])
      if not os.path.exists(imageFol):
          os.makedirs(imageFol)
      shutil.copy2(img, imageFol)