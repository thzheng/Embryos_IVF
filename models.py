import tensorflow as tf
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D
from tensorflow.python.keras import regularizers

def get_resnet50_baseline(output_classes):
  model = Sequential()
  model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=None))
  for layer in model.layers:
    layer.trainable = False
  model.add(Flatten())
  model.add(Dense(1024, activation='relu')) 
  model.add(Dropout(0.7))
  model.add(Dense(output_classes, activation='softmax')) 
  return model

def get_my_model(output_classes):
  model = Sequential()
  # conv 1, 2
  model.add(Conv2D(32, 3, input_shape=(224, 224, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None))
  model.add(Conv2D(32, 3, strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None))
  # pool 1
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
  # conv 3, 4
  model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None))
  model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None))
  # pool 2
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
  # conv 5, 6
  model.add(Conv2D(128, 3, strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None))
  model.add(Conv2D(128, 3, strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None))
  model.add(Conv2D(128, 3, strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None))
  # pool 3
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
  # flatten
  model.add(Flatten())
  # fc layers
  model.add(Dense(2048, activation='relu'))
  # model.add(Dropout(0.7))
  model.add(Dense(output_classes, activation='softmax')) 
  return model
