# copy data for unsupervised learning
import glob
import os
from random import sample
import shutil
minisets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
files = glob.glob("mnist_png/training/*/*.png")

for split in minisets:
  fol = "mnist_split/training_" + str(split)
  if not os.path.exists(fol):
    os.makedirs(fol)
  else:
    shutil.rmtree(fol)
    os.makedirs(fol)
  print(split)
  imgs = sample(files, int(split*len(files)))

  for img in imgs:
    imageFol = os.path.join(fol, os.path.dirname(img).split('/')[-1])
    if not os.path.exists(imageFol):
        os.makedirs(imageFol)
    shutil.copy2(img, imageFol)