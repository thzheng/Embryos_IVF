from models import *
from data_processing.data_processing import *

image_array, labels = get_resized_images(224, './CS231Snapshot')
assert len(image_array) == len(labels)
#model=get_resnet50_baseline(10)
#model.summary()

