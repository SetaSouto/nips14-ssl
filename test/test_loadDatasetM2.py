from hyperspectralData import HyperspectralData
import numpy as np

n_unlabeled = 1000
n_files = 10
n_max = 5000
x_l, y_l, x_u, y_u, valid_x, valid_y, test_x, test_y = HyperspectralData().load_dataset_m2(n_unlabeled=n_unlabeled,
                                                                                           n_files=n_files,
                                                                                           n_max=n_max)
print("Shapes:")
print("Train x labeled:", x_l.shape)
print("      y labaled:", y_l.shape)
print("Train x unlabeled:", x_u.shape)
print("      y unlabaled:", y_u.shape)
print("Valid x:", valid_x.shape)
print("      y:", valid_y.shape)
print("Test x:", test_x.shape)
print("     y:", test_y.shape)

def print_distribuiton(labels):
    total_per_class = np.sum(labels, axis=1)
    for i in range(100):
        if total_per_class[i]!=0:
            print("Label:", i, "Total:", total_per_class[i])

print('')
print("Train:")
print_distribuiton(y_l)
print('')
print("Valid:")
print_distribuiton(valid_y)
print('')
print("Test")
print_distribuiton(test_y)