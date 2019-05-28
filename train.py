from models import *
from data_processing.data_processing import *
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.datasets import fashion_mnist
import numpy as np
import skimage

(x_train_ori, y_train_ori), (x_test_ori, y_test_ori) = fashion_mnist.load_data()
x_train=[]
for i in range(len(x_train_ori)):
  x_train.append(skimage.transform.resize(x_train_ori[i], (64, 64, 1)))
x_train=np.asarray(x_train)
x_test=[]
for i in range(len(x_test_ori)):
  x_test.append(skimage.transform.resize(x_test_ori[i], (64, 64, 1)))
x_test=np.asarray(x_test)

y_train=to_categorical(y_train_ori, 10)
y_test=to_categorical(y_test_ori, 10)

# Split for train, val, test
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

# Train
model_type="DenseNet"
MyModel=get_densenet_baseline(10)
MyModel.summary()
MyAdam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0002)
MyModel.compile(optimizer = MyAdam, loss = "categorical_crossentropy", metrics = ["accuracy"])
MyEarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
MyMCP = ModelCheckpoint('.'+model_type+'_t.hdf5', save_best_only=True, monitor='val_loss', mode='min')
MyReducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')
MyModel.fit(x = x_train, y = y_train, epochs = 50, batch_size = 16, verbose=1, callbacks=[MyEarlyStopping, MyMCP, MyReducelr], validation_data=(x_val, y_val))

# Load the best check point
MyModel.load_weights(filepath='.'+model_type+'_t.hdf5')
score = MyModel.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
