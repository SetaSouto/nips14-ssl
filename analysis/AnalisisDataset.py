# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:24:17 2016

@author: fabian
"""

import h5py
import numpy as np

#If we want to work with unlabeled data
unlabeleds = True
#If we want to work with the mineralogy examples only
mineralogy = True

hdf5_file = "/media/fabian/IOMEGA HDD/Hyperspectral Data/Labeled HSI/ALH1599-17-labeled.hdf5"
with h5py.File(hdf5_file) as f:
    pixels = f['/hsimage/data'].value
    labels = f['/hsimage/labels'].value

if unlabeleds:
    labeled_indexes = labels > -1
else:
    labeled_indexes = labels > 0

pixels = pixels[labeled_indexes]
labels = labels[labeled_indexes]

if mineralogy:
    mineralogy_indexes = labels < 100
    pixels = pixels[mineralogy_indexes]
    labels = labels[mineralogy_indexes]
else:
    lithology_indexes = labels > 100
    pixels = pixels[lithology_indexes]
    labels = labels[lithology_indexes]

print("Pixels:")
print("Max:", pixels.max())
print("Min:", pixels.min())
print("Std:", pixels.std())
print("Shape:", pixels.shape)
print("Random example:")
print(pixels[np.random.randint(low=0, high=pixels.shape[0])])

print(" ")
print("Labels:")
print("Shape:", labels.shape)
print("Max:", labels.max())
print("Min:", labels.min())
print("Std:", labels.std())

print("Normalizando ...")

maxPix = pixels.max() + 1
pixels = pixels/maxPix

print("Pixels normalized:")
print("Max:", pixels.max())
print("Min:", pixels.min())
print("Std:", pixels.std())
print("Shape:", pixels.shape)
print("Random example:")
print(pixels[np.random.randint(low=0, high=pixels.shape[0])])
