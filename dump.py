from models import *
from data_processing.data_processing import *

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pickle

#model_type='my_model'
#model_type='resnet'
model_type='DenseNet'

#data_path='../EmbryoScopeAnnotatedData'
data_path='./Data'

# Resnet requires H and W >=224
if model_type=='DenseNet':
  data = get_resized_images(64, data_path, False)
else:
  data = get_resized_images(224, data_path, True)
pickle.dump( data, open( "dump.p", "wb" ) )
