from anglepy.hyperspectralData import HyperspectralData

train_x, train_y, valid_x, valid_y, test_x, test_y = HyperspectralData().load_numpy(50000)
