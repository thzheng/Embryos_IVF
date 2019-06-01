from models import *
from data_processing.data_processing import *

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

#model_type='my_model'
#model_type='resnet'
model_type='DenseNet'

# Resnet requires H and W >=224
if model_type=='DenseNet':
  x_ori, y_ori = get_resized_images(64, '../EmbryoScopeAnnotatedData')
else:
  x_ori, y_ori = get_resized_images(224, '../EmbryoScopeAnnotatedData')
assert len(x_ori) == len(y_ori)

print("labels and count:", np.unique(y_ori, return_counts= True))



# Prepare x
x_ori = np.asarray(x_ori, dtype='float32')

if model_type=='resnet':
  #  Change to Batch, Channel, W, H
  x_ori = np.swapaxes(x_ori, 1, 3)
  for n in range(x_ori.shape[0]):
    x_ori[n] = prepare_resnet50(x_ori[n])
  #  Change to Batch, W, H, Channel
  x_ori = np.swapaxes(x_ori, 1, 3)

# To one channel
if model_type=='DenseNet':
  x_ori = np.mean(x_ori, axis=3, keepdims=True)

#Prepare y
class_int = dict()
int_class = dict()

next_class = 1
for curr in y_ori:
  if curr!='tm' and curr!='teb':
    class_int[curr]=0
  else:
    if curr not in class_int:
      class_int[curr]=next_class
      int_class[next_class]=curr
      next_class+=1

num_class = next_class
for n in range(len(y_ori)):
  correct_label = class_int[y_ori[n]]
  y_ori[n] = np.zeros(num_class)
  y_ori[n][correct_label] = 1
y_ori = np.asarray(y_ori)

# Split for train, val, test
x_train, x_test, y_train, y_test = train_test_split(x_ori, y_ori, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

# Train
if model_type=='resnet':
  MyModel=get_resnet50_baseline(num_class)
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
  x = Dropout(0.7)(x)
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

