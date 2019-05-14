import tensorflow as tf
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, BatchNormalization

def get_resnet50_baseline(output_classes):
  model = Sequential()
  model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=None))
  model.add(Flatten())
  # model.add(BatchNormalization())
  model.add(Dense(output_classes, activation='softmax')) 
  return model
