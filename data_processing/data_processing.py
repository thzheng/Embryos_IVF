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
def yield_image(path):
  # Let's go through each folder and loop through all classes
  snapshot_basepath = path
  for folder_local in os.listdir(snapshot_basepath):
    print(folder_local)
    folder_full = os.path.join(snapshot_basepath, folder_local)
    for image_local in os.listdir(folder_full):
      image_full = os.path.join(folder_full, image_local)
      image_array = imread(image_full)[:,:,0:3]
      return image_array
"""
yield all images a resized size, all images are now squared, cropped from the middle
"""
def get_resized_images(img_size, path):
  for image_array in yield_image(path):
    # for each image, let's determine if they higher or wider, and crop at the middle
    shape0 = image_array.shape[0]
    shape1 = image_array.shape[1]
    if shape1 > shape0:
      d = shape1 - shape0
      image_array_cropped = image_array[:, d//2:-d//2, :]
    else:
      d = shape0 - shape1
      image_array_cropped = image_array[d//2:-d//2, :, :]
    print(image_array_cropped.shape)

    # Now let's resize!
    resized_image_array = np.array(Image.fromarray(image_array_cropped).resize((img_size, img_size)))
    return resized_image_array
