import numpy as np
import anglepy.ndict as ndict

# Path to the result's directory from the M1's training:
path = "results/hyper_50-(500, 500)_longrun/"
# Loads the parameters that has been training previously:
l1_v = ndict.loadz(path + 'v_best.ndict.tar.gz')
# Number of hidden nodes in the model:
n_h = (500, 500)
# Create the M1:
from anglepy.models.VAE_Z_X import VAE_Z_X

# We have to use the same hyperparameters from the training:
l1_model = VAE_Z_X(n_x=67 * 4, n_hidden_q=n_h, n_z=50, n_hidden_p=n_h, nonlinear_q='softplus', nonlinear_p='softplus',
                   type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1)

# Now we have to load the dataset that we wanna use.
from hyperspectralData import HyperspectralData

nsamples = 100
train_x, train_y, valid_x, valid_y, test_x, test_y = HyperspectralData().load_numpy(nsamples)


# Get the mean and the variance of the distribution learned to generate the z of the dataset.
def transform(v, _x):
    # Receives the values of the net and the data
    return l1_model.dist_qz['z'](*([_x] + list(v.values()) + [np.ones((1, _x.shape[1]))]))


# Extract the parameters of the distributions Z (features):
# For example, from the train train_x:
x_mean, x_logvar = transform(l1_v, train_x)
# And now, the features are obtained by sampling the distributions:
features = np.random.normal(x_mean, np.exp(0.5 * x_logvar))
