"""
Process the snapshot images
"""

from imageio import imread
from PIL import Image
import os
import numpy as np

"""
yield all images with the original H,W and always 3 channel
(the 4th channel is always 255 for a fullly saturated png, so we don't need it) 
"""
def yield_image(path, debug=0):
  # Let's go through each folder and loop through all classes
  snapshot_basepath = path
  for folder_local in os.listdir(snapshot_basepath):
    if debug > 0: print("Processing Folder:", folder_local)
    folder_full = os.path.join(snapshot_basepath, folder_local)
    for image_local in os.listdir(folder_full):
      image_full = os.path.join(folder_full, image_local)
      image_array = imread(image_full)[:,:,0:3]
      image_name_split = image_local.split('_')
      label_split = image_name_split[2].split('.')
      assert len(image_name_split) == 3 and len(label_split) == 2, "Image names should has length 3"
      #It needs to be splitted by ( because of names like t4(1).png
      yield (image_array,label_split[0].split('(')[0]) 
"""
yield all images a resized size, all images are now squared, cropped from the middle
"""
def get_resized_images(img_size, path, debug=0):
  all_images = []
  labels = []
  for image_array, label in yield_image(path, debug):
    # for each image, let's determine if they higher or wider, and crop at the middle
    shape0 = image_array.shape[0]
    shape1 = image_array.shape[1]
    if shape1 > shape0:
      d = shape1 - shape0
      image_array_cropped = image_array[:, d//2:-d//2, :]
    else:
      d = shape0 - shape1
      image_array_cropped = image_array[d//2:-d//2, :, :]
    if debug > 0: print(image_array_cropped.shape, label)

    # Now let's resize!
    resized_image_array = np.array(Image.fromarray(image_array_cropped).resize((img_size, img_size)))
    all_images.append(resized_image_array)
    labels.append(label)
  return all_images, labels
"""
Prepare image for resnet50
"""
def prepare_resnet50(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):  
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255.0 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data
