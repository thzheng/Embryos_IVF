from models import *
from data_processing.data_processing import *

from sklearn.model_selection import train_test_split
import numpy as np

x_ori, y_ori = get_resized_images(224, './CS231Snapshot')
assert len(x_ori) == len(y_ori)       

# Prepare x
x_ori = np.asarray(x_ori, dtype='float32')
#  Change to Batch, Channel, W, H
x_ori = np.swapaxes(x_ori, 1, 3)
for n in range(x_ori.shape[0]):
  x_ori[n] = prepare_resnet50(x_ori[n])

#Prepare y
class_int = dict()
int_class = dict()
next_class = 0
for curr in y_ori:
  if curr not in class_int:
    class_int[curr]=next_class
    int_class[next_class]=curr
    next_class+=1

y_ori = np.asarray(y_ori)

# Split for train, val, test
x_train, x_test, y_train, y_test = train_test_split(x_ori, y_ori, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

#model=get_resnet50_baseline(10)
#model.summary()

