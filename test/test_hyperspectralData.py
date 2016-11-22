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
