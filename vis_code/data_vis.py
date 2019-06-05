import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import random


file = open('dump.p','rb') 
input_dict = pickle.load(open('dump.p','rb'))
# print((input_dict['3_2'])[0])

print (len(input_dict))

X = []
y = []

stride = 50

label_dict = {}

def processLabel(str):
	if str == 'tEB':
		return 1
	elif str == 'tM':
		return 2
	return 0

for k, v in input_dict.items():
	for data in v:
		new_data = data[0].flatten()
		lb = data[1]
		# store in dictionary
		if lb not in label_dict:
			label_dict[lb] = list()
		label_dict[lb].append(new_data)


new_X = []
new_Y = []
# take random samples of 100 from each class
for k, v in label_dict.items():
	np.random.shuffle(v)
	print (k)
	new_X += v[:stride]
	new_Y += [k] * stride


print (new_X[0])
print (new_Y[0])

X_array = np.concatenate(new_X, axis = 0)

# first we just use 3 classes, each with at most 100 images.

print("Data num is : " + str(len(X)))


color = ['black', 'red', 'green', 'blue', 'gold', 
		'purple','lavender', 'orchid', 'teal', 'yellow', '#800000', 'aqua', 'grey', 'brown', '#aaffc3']

label = ['0', '1', '2']
label_2 = list(label_dict.keys())
print (label_2)

print(new_X[0])

fig, ax = plt.subplots()

X_PCA = PCA(n_components=2).fit_transform(np.matrix(new_X))
X_PCA = TSNE(n_components=2).fit_transform(np.matrix(new_X))
print ("The {} dims images have be projected to {} D space".format(x[0].shape, X_PCA.shape[1]))

# X_PCA = PCA(n_components=2).fit_transform(X)

# class_num = 3
for i in range(len(label_2)):
	ax.scatter(X_PCA[i*stride:(i+1)*stride,0], X_PCA[i*stride:(i+1)*stride:,1], c=color[i], label=label_2[i], alpha=0.3, edgecolors='none')
plt.show()

