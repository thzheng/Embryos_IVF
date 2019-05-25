from models import *
from data_processing.data_processing import *
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.datasets import fashion_mnist
import numpy as np
import skimage

(x_train_ori, y_train), (x_test_ori, y_test) = fashion_mnist.load_data()
x_train=[]
for i in range(len(x_train_ori)):
  x_train.append(skimage.transform.resize(x_train_ori[i], (64, 64)))
x_train=np.asarray(x_train)
print(x_train_ori.shape)
print(x_train.shape)


# Split for train, val, test
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)


"""
# Train
if model_type=='resnet':
  MyModel=get_resnet50_baseline(num_class)
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
"""
