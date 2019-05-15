from models import *
from data_processing.data_processing import *

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

model_type='my_model'
# model_type='resnet'

x_ori, y_ori = get_resized_images(224, './CS231Snapshot')
assert len(x_ori) == len(y_ori)       

# Prepare x
x_ori = np.asarray(x_ori, dtype='float32')

if model_type=='resnet':
  #  Change to Batch, Channel, W, H
  x_ori = np.swapaxes(x_ori, 1, 3)
  for n in range(x_ori.shape[0]):
    x_ori[n] = prepare_resnet50(x_ori[n])
  #  Change to Batch, W, H, Channel
  x_ori = np.swapaxes(x_ori, 1, 3)

#Prepare y
class_int = dict()
int_class = dict()
next_class = 0
for curr in y_ori:
  if curr not in class_int:
    class_int[curr]=next_class
    int_class[next_class]=curr
    next_class+=1
for n in range(len(y_ori)):
  correct_label = class_int[y_ori[n]]
  y_ori[n] = np.zeros(15)
  y_ori[n][correct_label] = 1
y_ori = np.asarray(y_ori)

# Split for train, val, test
x_train, x_test, y_train, y_test = train_test_split(x_ori, y_ori, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

# Train
if model_type=='resnet':
  MyModel=get_resnet50_baseline(15)
elif model_type=='my_model':
  MyModel=get_my_model(15)

MyModel.summary()
MyAdam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0002)
MyModel.compile(optimizer = MyAdam, loss = "categorical_crossentropy", metrics = ["accuracy"])
MyEarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
MyMCP = ModelCheckpoint('.'+model_type+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
MyReducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')
MyModel.fit(x = x_train, y = y_train, epochs = 50, batch_size = 16, verbose=1, callbacks=[MyEarlyStopping, MyMCP, MyReducelr], validation_data=(x_val, y_val))

# Load the best check point
MyModel.load_weights(filepath='.'+model_type+'.hdf5')
score = MyModel.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

