# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:53:48 2016

@author: fabian
"""
import numpy as np
import anglepy.data.mnist as mnist

size = 28
train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy(size)

print("--------------------")
print("Shapes of the numpy arrays.")
print(" ")
print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)
print("valid_x shape:", valid_x.shape)
print("valid_y shape:", valid_y.shape)
print("test_x shape:", test_x.shape)
print("test_y shape:", test_y.shape)
print("--------------------")
print("Descriptions of the datasets:")
print(" ")
print("train_x:")
print("Max:", train_x.max())
print("Min:", train_x.min())
print("Std:", train_x.std())
print("Random example:", train_x[np.random.randint(low=0, high=train_x.shape[0]-1)])
print(" ")
print("train_y:")
print("Max:", train_y.max())
print("Min:", train_y.min())
print("Random example:", train_y[np.random.randint(low=0, high=train_y.shape[0]-1)])

# Ahora para el aprendizaje semi supervisado:

train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy_split(size, binarize_y=True)
# create labeled/unlabeled split in training set
x_l, y_l, x_u, y_u = mnist.create_semisupervised(train_x, train_y, 100)

print("--------------------")
print("Shape of the datasets.")
print(" ")
print("Train x labeled:", x_l.shape)
print("Train y labeled:", y_l.shape)
print("Train x unlabeled:", x_u.shape)
print("Train y unlabeled:", y_u.shape)
print(" ")
print("Descriptions:")
print(" ")
print("X labeled:")
print("Max:", x_l.max())
print("Min:", x_l.min())
print("Std:", x_l.std())
#print("Random example", x_l[: , np.random.randint(low=0, high=x_l.shape[1]-1)])
print(" ")
print("Y Labeled")
print("Max:", y_l.max())
print("Min:", y_l.min())
print("Std:", y_l.std())
print("Random example:", y_l[:, np.random.randint(low=0, high=y_l.shape[1]-1)])
print(" ")
print("Y Unlabeled")
print("Max:", y_u.max())
print("Min:", y_u.min())
print("Std:", y_u.std())
print("Random example", y_u[:, np.random.randint(low=0, high=y_u.shape[1]-1)])
