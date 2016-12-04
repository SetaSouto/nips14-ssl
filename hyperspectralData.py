# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:46:17 2016

@author: fabian
"""

import h5py
import numpy as np
import os


class HyperspectralData:
    def __init__(self,
                 path="/home/fabian/DataScience/SemiSupervisedLearning/nips14-ssl/data/Hyperspectral Data/Labeled HSI/"):
        self.path = path

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

    def get_filenames(self, n_filenames):
        """
        Returns a list with n_filenames filenames in the directory indicated by path.
        :return: List with files in self.path.
        """
        return os.listdir(self.get_path())[:n_filenames]

    def load_pixels_labels(self, n_files=1, labeled=False, mineralogy=True):
        """
        Returns a numpy array with the pixels and another numpy array with the labels.

        :param file: File to be loaded.
        :return: Numpy array with pixels and a numpy array with the labels.
        """
        if n_files<1:
            raise Exception()
        # First file:
        pixels, labels = self.load_pixels_labels_by_filename(self.get_filenames(1)[0], labeled, mineralogy)
        # The others files:
        for filename in self.get_filenames(n_filenames=n_files)[1:]:
            new_pixels, new_labels = self.load_pixels_labels_by_filename(filename, labeled, mineralogy)
            pixels = np.concatenate((pixels, new_pixels), axis=0)
            labels = np.concatenate((labels, new_labels), axis=0)

        return pixels, labels

    def load_pixels_labels_by_filename(self, filename, labeled=False, mineralogy=True):
        """
        Returns the values of the pixels and the labels for a given file.
        :param filename: Filename to load.
        :return: Pixels, labels.
        """
        hdf5_file = self.get_path() + filename
        with h5py.File(hdf5_file) as f:
            pixels = f['/hsimage/data'].value
            labels = f['/hsimage/labels'].value
        if labeled:
            indexes = labels > 0
            pixels = pixels[indexes]
            labels = labels[indexes]
        if mineralogy:
            indexes = labels < 100
            pixels = pixels[indexes]
            labels = labels[indexes]
        return pixels, labels

    def load_numpy(self, n_train, n_valid=10000, n_test=10000, labeled_and_unlabeled=True, labeled=True,
                   mineralogy=True,
                   n_files=1):
        """Returns numpy arrays with the data.

        :param labeled_and_unlabeled: Indicates if unlabeled examples must be used with the labeled ones.
        :param labeled: Indicates if in case that we want to use only labeled samples or unlabeled samples.
        :param mineralogy: Indicates if mineralogy data must be used. If not, lithology examples are used instead.
        :param n_train: Numbers of examples for training.
        :param n_valid: Numbers of examples for validation.
        :param n_test: Numbers of examples for testing.
        :param n_files: From how many files we want to extract te samples.
        """

        pixels, labels = self.load_pixels_labels(n_files)

        if labeled_and_unlabeled:
            labeled_indexes = labels > -1
        elif labeled:
            labeled_indexes = labels > 0
        else:
            labeled_indexes = labels == 0

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

    def to_one_hot(self, y, n_classes):
        """
        Transform to one hot encoding an array with labels.

        :param y: Labels.
        :return: Labels in one hot encoding.
        """

        n_labels = y.shape[0]
        ret = np.zeros((n_classes, n_labels))
        ret[y, np.arange(n_labels)] = 1
        # But this makes 1 whenever y is zero, so we must put zero in all the first row:
        ret[0, :] = 0
        return ret

    def get_labeled_numpy(self, n_train, n_valid, n_test, n_files=1):
        """
        Returns only labeled samples.
        :param n_train: Number of samples for training.
        :param n_valid: Number of samples for validation.
        :param n_test: Number of samples for testing.
        :param n_files: Number of files to check to extract the amount of samples.
        :return: train_x, train_y, valid_x, valid_y, test_x, test_y
        """
        return self.load_numpy(n_train, n_valid, n_test, labeled_and_unlabeled=False, labeled=True)

    def get_unlabeled_numpy(self, n_train, n_valid, n_test):
        """
        Returns only unlabeled samples.
        :param n_train: Number of samples for training.
        :param n_valid: Number of samples for validation.
        :param n_test: Number of samples for testing.
        :return: train_x, train_y, valid_x, valid_y, test_x, test_y
        """
        return self.load_numpy(n_train, n_valid, n_test, labeled_and_unlabeled=False, labeled=False)

    def load_dataset_m2(self, n_unlabeled, n_files=10, n_max=5000, n_classes=100):
        """
        A new way to load the datasets, because the old way doesn't give a uniform distribuiton of labels.
        By this way, we load n_files files, and try to keep no more that n_max samples by class, so the labels that
        are the majority in the files are not the majority in the return of this function.

        :param n_unlabeled: Number of unlabeled samples to load.
        :param n_files: Number of files where the data will be loaded.
        :param n_max: Number of max samples per class.
        :param n_classes: Number of classes in the data, it is usefull to pass the labels to one hot encoding.
        :return: train_x_labeled, train_y_labeled, train_x_unlabeled, train_y_unlabeled, valid_x, valid_y, test_x, test_y
        """
        print("---------------")
        print("Loading samples.")
        pixels, labels = self.load_pixels_labels(n_files=n_files, labeled=True, mineralogy=True)

        print("Formatting data.")
        # The shape is (n_examples, data), we need (data, n_examples)
        pixels = np.transpose(pixels)
        # Normalize data: Max value can't be more than one.
        pixels = pixels / (pixels.max() + 1.0)

        # Iterate over the labels, delete samples if we have more than 5000
        for label in range(labels.max() + 1):
            # How many of them are:
            indexes = labels == label
            n = labels[indexes].shape[0]
            if n > n_max:
                indexes_to_delete = np.random.choice(indexes, size=(n - n_max), replace=False)
                pixels = np.delete(pixels, indexes_to_delete, axis=1)
                labels = np.delete(labels, indexes_to_delete, axis=0)

        # To one hot encoding
        labels = self.to_one_hot(labels, n_classes)
        # Select the rows randomnly
        n_total = pixels.shape[1]
        print("Number of labeled samples loaded:", n_total)
        columns = np.random.choice(n_total, size=n_total, replace=False)
        n_train = n_total/2 # Half is for training
        n_rest = n_total/4  # Valid and test have the other half
        # Indexes
        ind_train = columns[0:n_train]
        ind_valid = columns[n_train:(n_train + n_rest)]
        ind_test = columns[(n_train + n_rest): (2*n_rest)]

        # Select data:
        train_xl = pixels[:, ind_train]
        train_yl = labels[:, ind_train]

        valid_x = pixels[:, ind_valid]
        valid_y = labels[:, ind_valid]

        test_x = pixels[:, ind_test]
        test_y = labels[:, ind_test]

        # Now, load unlabeled samples:
        print("Loading unlabeled samples.")
        x_u, y_u, _, _, _, _ = self.get_unlabeled_numpy(n_unlabeled, 1, 1)
        # To one hot encoding:
        y_u = self.to_one_hot(y_u, n_classes)

        return train_xl, train_yl, x_u, y_u, valid_x, valid_y, test_x, test_y



# Class for raise a match error between the labels and data.
class MatchError(Exception):
    def __init__(self, message):
        self.message = message
