from data_processing.data_processing import *
import numpy as np
import matplotlib.pyplot as plt

x_ori, y_ori = get_resized_images(224, './CS231Snapshot')
plt.hist(y_ori)
plt.xlabel('label')
plt.ylabel('count')
plt.title('Histogram of Current Classes Count')
plt.show()
assert len(x_ori) == len(y_ori)

print("labels and count:", np.unique(y_ori, return_counts= True))