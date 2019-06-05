from models import *
from data_processing.data_processing import *

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

#model_type='my_model'
#model_type='DenseNet_full'
model_type='DenseNet'

data=pickle.load( open( "dump.p", "rb" ) )

x_ori=[]
y_ori=[]
tag=[]
for well in data:
  for i in range(len(data[well])):
    x_ori.append(data[well][i][0])
    y_ori.append(data[well][i][1])
    tag.append(well)
assert len(x_ori) == len(y_ori)
assert len(x_ori) == len(tag)

#Prepare y
class_int = dict()
int_class = dict()

next_class = 0
#next_class = 1
for curr in y_ori:
  #"""
  if curr not in class_int:
    class_int[curr]=next_class
    int_class[next_class]=curr
    next_class+=1
  #"""
  """
  if curr!='tM' and curr!='tEB':
    class_int[curr]=0
  else:
    if curr not in class_int:
      class_int[curr]=next_class
      int_class[next_class]=curr
      next_class+=1
  """

num_class = next_class
print("num_class: ", num_class)
for n in range(len(y_ori)):
  correct_label = class_int[y_ori[n]]
  y_ori[n] = np.zeros(num_class)
  y_ori[n][correct_label] = 1

# Split for train, val, test, split by well
x_train=[]
y_train=[]
x_val=[]
y_val=[]
x_test=[]
y_test=[]
for i in range(len(x_ori)):
  if tag[i]=='9_4' or tag[i]=='9_5' or tag[i]=='9_6':
    x_val.append(x_ori[i])
    y_val.append(y_ori[i])
  elif tag[i]=='8_2' or tag[i]=='8_3' or tag[i]=='8_4':
    x_test.append(x_ori[i])
    y_test.append(y_ori[i])
  else:
    x_train.append(x_ori[i])
    y_train.append(y_ori[i])

x_train = np.asarray(x_train, dtype='float32')
x_train = x_train[..., None]
y_train = np.asarray(y_train)
x_val = np.asarray(x_val, dtype='float32')
x_val = x_val[..., None]
y_val = np.asarray(y_val)
x_test = np.asarray(x_test, dtype='float32')
x_test = x_test[..., None]
y_test = np.asarray(y_test)



# Train
if model_type=='resnet':
  MyModel=get_resnet50_baseline(num_class)
elif model_type=='DenseNet_full':
  MyModel=get_densenet_baseline(num_class)
elif model_type=='DenseNet':
  TModel=get_densenet_baseline(10)
  TModel.load_weights(filepath='.'+model_type+'_t.hdf5')
  print("Transfer from Model:")
  TModel.summary()
  XModel=Model(TModel.inputs, TModel.layers[-3].output)
  XModel.set_weights(TModel.get_weights())
  for layer in XModel.layers:
    layer.trainable = False
  XModel.trainable = False
  print("Model before FC:")
  XModel.summary()
  x = XModel.layers[-1].output
  x = Dropout(0.8)(x)
  #x = Dense(2048, activation='relu')(x)
  x = Dense(num_class, activation='softmax')(x)
  MyModel = Model(XModel.inputs, x)
elif model_type=='my_model':
  MyModel=get_my_model(num_class)

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

