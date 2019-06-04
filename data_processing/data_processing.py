"""
Process the snapshot images
"""

from imageio import imread
from PIL import Image
import os
import numpy as np
import skimage
from .data_processing2 import YieldImage as NewYieldImage
from collections import defaultdict

"""
yield all images with the original H,W and always 3 channel
(the 4th channel is always 255 for a fully saturated png, so we don't need it) 
"""
def yield_image(path, debug=0):
  # Let's go through each folder and loop through all classes
  snapshot_basepath = path
  for folder_local in os.listdir(snapshot_basepath):
    if folder_local[0] == '.': continue
    if debug > 0: print("Processing Folder:", folder_local)
    folder_full = os.path.join(snapshot_basepath, folder_local)
    if not os.path.isdir(folder_full): continue
    for content_name in os.listdir(folder_full):
      if content_name[0] == '.': continue
      full_path = os.path.join(folder_full, content_name)
      if os.path.isfile(full_path):
        # It's an image! This is the original screenshots that we took from the videos
        image_local = content_name
        image_full = full_path
        if debug > 0: print("Reading File 1", image_full)
        image_array = imread(image_full)[:,:,0:3]
        image_name_split = image_local.split('_')
        label_split = image_name_split[2].split('.')
        assert len(image_name_split) == 3 and len(label_split) == 2, "Image names should has length 3"
        #It needs to be splitted by ( because of names like t4(1).png
        yield (image_array,label_split[0].split('(')[0])
      else:
        # This is the newly added images labeled with well_#/label/image.jpg
        if not os.path.isdir(full_path): continue
        for label in os.listdir(full_path):
          if label[0] == '.': continue
          full_path_with_label = os.path.join(full_path, label)
          for image_local in os.listdir(full_path_with_label):
            if image_local[0] == '.': continue
            # It's an image! 
            image_full = os.path.join(full_path_with_label, image_local)
            if debug > 0: print("Reading File 2", full_path, label, image_local, image_full)
            image_array = np.swapaxes(np.swapaxes([imread(image_full),]*3, 0, 2), 0 ,1)
            if debug > 0: print("File 2 Shape", image_array.shape)
            # image_array = imread(image_full)[:,:,0:3]
            yield (image_array, label)

"""
yield all images a resized size, all images are now squared, cropped from the middle
"""
def get_resized_images(img_size, path, pad_to_3channel, debug=0):
  ret = defaultdict(list)
  for image_array, label, folder_num, well_num in NewYieldImage(path, pad_to_3channel, debug):
    # for each image, let's determine if they higher or wider, and crop at the middle
    shape0 = image_array.shape[0]
    shape1 = image_array.shape[1]
    if shape1 > shape0:
      d = shape1 - shape0
      image_array_cropped = image_array[:, d//2:-d//2, :]
    elif shape1 < shape0:
      d = shape0 - shape1
      image_array_cropped = image_array[d//2:-d//2, :, :]
    else: image_array_cropped = image_array
    if debug > 0: print(image_array_cropped.shape, label)

    # Now let's resize!
    resized_image_array = np.array(Image.fromarray(image_array_cropped).resize((img_size, img_size)))
    well_id = str(folder_num) + "_" + str(well_num)
    ret[well_id].append((resized_image_array, label))
  return ret

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
"""
Resize a single channel image
"""
def resize_img(img, x, y):
    return skimage.transform.resize(x_train_ori[i], (x, y))

# get_resized_images(224, '../../EmbryoScopeAnnotatedData', 1)
