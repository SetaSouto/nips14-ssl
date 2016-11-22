# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:46:17 2016

@author: fabian
"""

import h5py
import numpy as np


class HyperspectralData:
    def __init__(self, path="/home/fabian/DataScience/SemiSupervisedLearning/nips14-ssl/data/Hyperspectral Data/Labeled HSI/"):
        self.path = path

    def load_pixels_labels(self, file="ALH1599-17-labeled.hdf5"):
        """
        Returns a numpy array with the pixels and another numpy array with the labels.

        :param file: File to be loaded.
        :return: Numpy array with pixels and a numpy array with the labels.
        """

        hdf5_file = self.get_path() + file

        with h5py.File(hdf5_file) as f:
            pixels = f['/hsimage/data'].value
            labels = f['/hsimage/labels'].value

        return pixels, labels

    def load_numpy(self, n_train, n_valid=10000, n_test=10000, unlabeled=True, mineralogy=True,
                   file="ALH1599-17-labeled.hdf5"):
        """Returns numpy arrays with the data.

        :param unlabeled: Indicates if unlabeled examples must be used.
        :param mineralogy: Indicates if mineralogy data must be used. If not, lithology examples are used instead.
        :param n_train: Numbers of examples for training.
        :param n_valid: Numbers of examples for validation.
        :param n_test: Numbers of examples for testing.
        :param file: File to be loaded.
        """

        pixels, labels = self.load_pixels_labels(file)

        if unlabeled:
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

        # The shape is (n_examples, data), we need (data, n_examples)
        pixels = np.transpose(pixels)

        # Normalize data: Max value can't be more than one.
        pixels = pixels / (pixels.max() + 1.0)

        # Now, select randomly the datasets

        max_row = pixels.shape[1]
        total_to_be_selected = n_train + n_test + n_valid

        # Generate a random narray with the number of rows to be selected without repetition.
        rows = np.random.choice(max_row, size=total_to_be_selected, replace=False)

        valid_rows = rows[0: n_valid]
        test_rows = rows[n_valid: (n_valid + n_test)]
        train_rows = rows[(n_valid + n_test): total_to_be_selected]

        train_x = pixels[:, train_rows]
        train_y = labels[train_rows]

        valid_x = pixels[:, valid_rows]
        valid_y = labels[valid_rows]

        test_x = pixels[:, test_rows]
        test_y = labels[test_rows]

        # Test: We need to check that the labels are not corrupt, for example
        # taking the label of another sample.
        data1 = pixels[:, rows[n_valid]]  # First sample of the test dataset.
        label1 = labels[rows[n_valid]]  # The correspondent label
        data2 = test_x[:, 0]  # First sample of test dataset.
        label2 = test_y[0]  # The correspondent label.
        if (data1 != data2).all():
            raise MatchError("Data does not match.")
        if (data1 == data2).all() and (label1 != label2):
            raise MatchError("Labels does not match.")

        # If we don't have errors, return.
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def get_path(self):
        """
        Returns the path where are the files.
        """
        return self.path

    def set_path(self, new_path):
        """
        Sets the path where are the files.
        """
        self.path = new_path

    def split_data(self, x, y):
        """
        Receives a dataset with some unlabeled samples. Returns the dataset divided in those that has labels and those
        who don't.

        :param x: Dataset.
        :param y: Labels.
        :return: Dataset divided in labeled and unlabeled.
        """
        labeled_indexes = y > 0
        unlabeled_indexes = y == 0

        # Select columns, keep the rows:
        x_labeled = x[:, labeled_indexes]
        y_labeled = y[labeled_indexes]
        x_unlabeled = x[:, unlabeled_indexes]
        y_unlabeled = y[unlabeled_indexes]

        return x_labeled, y_labeled, x_unlabeled, y_unlabeled




# Class for raise a match error between the labels and data.
class MatchError(Exception):
    def __init__(self, message):
        self.message = message
