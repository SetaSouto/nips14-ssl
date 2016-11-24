from unittest import TestCase
from hyperspectralData import HyperspectralData


class TestHyperspectralData(TestCase):
    n_train = 200
    n_valid = 100
    n_test = 150
    train_x, train_y, valid_x, valid_y, test_x, test_y = HyperspectralData().load_numpy(n_train=n_train,
                                                                                        n_valid=n_valid,
                                                                                        n_test=n_test)

    def test_load_numpy(self):
        # Loaded the number of examples that we request:
        self.assertEqual(self.train_x.shape[1], self.n_train)
        self.assertEqual(self.valid_x.shape[1], self.n_valid)
        self.assertEqual(self.test_x.shape[1], self.n_test)

        # Labels has the same size that the dataset:
        self.assertEqual(self.train_x.shape[1], self.train_y.shape[0])
        self.assertEqual(self.valid_x.shape[1], self.valid_y.shape[0])
        self.assertEqual(self.test_x.shape[1], self.test_y.shape[0])

    def test_split_data(self):
        x_labeled, y_labeled, x_unlabeled, y_unlabeled = HyperspectralData().split_data(self.train_x, self.train_y)

        self.assertEqual(x_labeled.shape[1], y_labeled.shape[0])
        self.assertEqual(x_unlabeled.shape[1], y_unlabeled.shape[0])
        # Labeled + unlabeled = total:
        self.assertEqual(self.n_train, x_labeled.shape[1]+x_unlabeled.shape[1])

    def test_to_one_hot(self):
        # In mineralogy are 100 classes.
        n_classes = 100
        y_one_hot = HyperspectralData().to_one_hot(self.train_y, n_classes)
        # It must have a row for each class:
        self.assertEqual(n_classes, y_one_hot.shape[0])
        # It must have a column for each label:
        self.assertEqual(self.train_y.shape[0], y_one_hot.shape[1])

        # It must have a 1 in the row indicated by train_y:
        for i in range(y_one_hot.shape[1]):         # Iterate over the columns
            if (self.train_y[i]==0):                # If it doesn't have a label the entire column must be zero
                for j in range(y_one_hot.shape[0]): # Iterate over the rows of the column i
                    self.assertEqual(0., y_one_hot[j, i])
            else:
                self.assertEqual(1., y_one_hot[self.train_y[i], i])

