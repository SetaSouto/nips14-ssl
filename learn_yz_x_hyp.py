import os
import time

import anglepy.ndict as ndict
import numpy as np
import theano
import theano.tensor as T
from adam import AdaM


def main(n_passes, n_hidden, seed, alpha, n_minibatches, n_labeled, n_unlabeled, n_classes):
    """
    Learn a variational auto-encoder with generative model p(x,y,z)=p(y)p(z)p(x|y,z)
    And where 'x' is always observed and 'y' is _sometimes_ observed (hence semi-supervised).
    We're going to use q(y|x) as a classification model.
    """

    # Create the directory for the log and outputs.
    logdir = 'results/learn_yz_x_hyp' + '-' + str(int(time.time())) + '/'
    if not os.path.exists(logdir): os.makedirs(logdir)
    print("---------------")
    print('Logdir:', logdir)

    # Feed with the seed:
    np.random.seed(seed)

    # Load model for feature extraction
    path = 'results/hyper_50-(500, 500)_longrun/'
    # Load the parameters of the model that has been trained previously:
    l1_v = ndict.loadz(path + 'v_best.ndict.tar.gz')
    l1_w = ndict.loadz(path + 'w_best.ndict.tar.gz')

    # Same hyperparameters that we use for training M1:

    # Number of hidden nodes in the model:
    n_h = (500, 500)
    # Size of our feature vector:
    n_x = 67 * 4
    # Number of latent variables:
    n_z = 50
    nonlinear = 'softplus'
    type_px = 'bernoulli'
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'


    # Create the M1:
    from anglepy.models.VAE_Z_X import VAE_Z_X
    l1_model = VAE_Z_X(n_x=n_x, n_hidden_q=n_h, n_z=n_z, n_hidden_p=n_h, nonlinear_q=nonlinear,
                       nonlinear_p=nonlinear, type_px=type_px, type_qz=type_qz, type_pz=type_pz,
                       prior_sd=1)

    # Load dataset:
    x_l, y_l, x_u, y_u, valid_x, valid_y, test_x, test_y = load_dataset(n_labeled, n_unlabeled, n_classes)


    # Extract features

    def transform(v, _x):
        # Get the mean and the variance of the distribution learned to generate the z of the dataset.
        return l1_model.dist_qz['z'](*([_x] + list(v.values()) + [np.ones((1, _x.shape[1]))]))

    # 3. Extract features
    x_mean_u, x_logvar_u = transform(l1_v, x_u)
    x_mean_l, x_logvar_l = transform(l1_v, x_l)
    x_unlabeled = {'mean': x_mean_u, 'logvar': x_logvar_u, 'y': y_u}
    x_labeled = {'mean': x_mean_l, 'logvar': x_logvar_l, 'y': y_l}

    valid_x, _ = transform(l1_v, valid_x)
    test_x, _ = transform(l1_v, test_x)

    # Copied from learn_yz_x_ss:
    n_x = l1_w[b'w0'].shape[1]
    n_y = n_classes
    type_pz = 'gaussianmarg'
    type_px = 'gaussian'
    nonlinear = 'softplus'

    # Init VAE model p(x,y,z)
    from anglepy.models.VAE_YZ_X import VAE_YZ_X
    uniform_y = True
    model = VAE_YZ_X(n_x, n_y, n_hidden, n_z, n_hidden, nonlinear, nonlinear, type_px, type_qz=type_qz,
                     type_pz=type_pz, prior_sd=1, uniform_y=uniform_y)
    v, w = model.init_w(1e-3)

    # Init q(y|x) model
    from anglepy.models.MLP_Categorical import MLP_Categorical
    n_units = [n_x] + list(n_hidden) + [n_y]
    model_qy = MLP_Categorical(n_units=n_units, prior_sd=1, nonlinearity=nonlinear)
    u = model_qy.init_w(1e-3)

    write_headers(logdir)

    # Progress hook
    t0 = time.time()

    def hook(step, u, v, w, ll):

        print("---------------")
        print("Current results:")
        print(" ")

        # Get classification error of validation and test sets
        def error(dataset_x, dataset_y):
            _, _, _z = model_qy.gen_xz(u, {'x': dataset_x}, {})
            print("  Predictions:", np.argmax(_z['py'], axis=0)[0:20])
            print("  Real:       ", np.argmax(dataset_y, axis=0)[0:20])
            return np.sum(np.argmax(_z['py'], axis=0) != np.argmax(dataset_y, axis=0)) / (0.0 + dataset_y.shape[1])

        print("Validset:")
        valid_error = error(valid_x, valid_y)
        print("Testset:")
        test_error = error(test_x, test_y)

        # Save variables
        ndict.savez(u, logdir + 'u')
        ndict.savez(v, logdir + 'v')
        ndict.savez(w, logdir + 'w')

        time_elapsed = time.time() - t0

        # This will be showing the current results and write them in a file:
        with open(logdir + 'AA_results.txt', 'a') as file:
            file.write(str(step) + ',' + str(time_elapsed) + ',' + str(valid_error) + ',' + str(test_error) + '\n')

        print("Step:", step)
        print("Time elapsed:", time_elapsed)
        print("Validset error:", valid_error)
        print("Testset error:", test_error)
        print("LogLikelihood:", ll)

        return valid_error

    # Optimize
    result = optim_vae_ss_adam(alpha, model_qy, model, x_labeled, x_unlabeled, n_y, u, v, w,
                               n_minibatches=n_minibatches, n_passes=n_passes, hook=hook)

    return result


def optim_vae_ss_adam(alpha, model_qy, model, x_labeled, x_unlabeled, n_y, u_init, v_init, w_init, n_minibatches,
                      n_passes, hook, n_reset=20, resample_keepmem=False, display=0):
    # Shuffle datasets
    ndict.shuffleCols(x_labeled)
    ndict.shuffleCols(x_unlabeled)

    # create minibatches
    minibatches = []

    n_labeled = next(iter(x_labeled.values())).shape[1]
    n_batch_l = n_labeled / n_minibatches
    if (n_labeled % n_batch_l) != 0: raise Exception()

    n_unlabeled = next(iter(x_unlabeled.values())).shape[1]
    n_batch_u = n_unlabeled / n_minibatches
    if (n_unlabeled % n_batch_u) != 0: raise Exception()

    n_tot = n_labeled + n_unlabeled

    # Divide into minibatches
    def make_minibatch(i):
        _x_labeled = ndict.getCols(x_labeled, i * n_batch_l, (i + 1) * n_batch_l)
        _x_unlabeled = ndict.getCols(x_unlabeled, i * n_batch_u, (i + 1) * n_batch_u)
        return [i, _x_labeled, _x_unlabeled]

    for i in range(n_minibatches):
        minibatches.append(make_minibatch(i))

    # For integrating-out approach
    L_inner = T.dmatrix()
    L_unlabeled = T.dot(np.ones((1, n_y)), model_qy.p * (L_inner - T.log(model_qy.p)))
    grad_L_unlabeled = T.grad(L_unlabeled.sum(), list(model_qy.var_w.values()))
    f_du = theano.function([model_qy.var_x['x']] + list(model_qy.var_w.values()) + [model_qy.var_A, L_inner],
                           [L_unlabeled] + grad_L_unlabeled)

    # Some statistics
    L = [0.]
    n_L = [0]

    def f_df(w, minibatch):

        u = w['u']
        v = w['v']
        w = w['w']

        i_minibatch = minibatch[0]
        _x_l = minibatch[1]  # labeled
        x_minibatch_l = {'x': np.random.normal(_x_l['mean'], np.exp(0.5 * _x_l['logvar'])), 'y': _x_l['y']}
        eps_minibatch_l = model.gen_eps(n_batch_l)

        _x_u = minibatch[2]  # unlabeled
        x_minibatch_u = {'x': np.random.normal(_x_u['mean'], np.exp(0.5 * _x_u['logvar'])), 'y': _x_u['y']}
        eps_minibatch_u = [model.gen_eps(n_batch_u) for i in range(n_y)]

        # === Get gradient for labeled data
        # gradient of -KL(q(z|y,x) ~p(x,y) || p(x,y,z))
        logpx, logpz, logqz, gv_labeled, gw_labeled = model.dL_dw(v, w, x_minibatch_l, eps_minibatch_l)
        # gradient of classification error E_{~p(x,y)}[q(y|x)]
        logqy, _, gu_labeled, _ = model_qy.dlogpxz_dwz(u, x_minibatch_l, {})

        # Reweight gu_labeled and logqy
        # beta = alpha / (1.-alpha) * (1. * n_unlabeled / n_labeled) #old
        beta = alpha * (1. * n_tot / n_labeled)
        for i in u: gu_labeled[i] *= beta
        logqy *= beta

        L_labeled = logpx + logpz - logqz + logqy

        # === Get gradient for unlabeled data
        # -KL(q(z|x,y)q(y|x) ~p(x) || p(x,y,z))
        # Approach where outer expectation (over q(z|x,y)) is taken as explicit sum (instead of sampling)
        u = ndict.ordered(u)
        py = model_qy.dist_px['y'](*([x_minibatch_u['x']] + list(u.values()) + [np.ones((1, n_batch_u))]))

        if True:
            # Original
            _L = np.zeros((n_y, n_batch_u))
            gv_unlabeled = {i: 0 for i in v}
            gw_unlabeled = {i: 0 for i in w}
            for label in range(n_y):
                new_y = np.zeros((n_y, n_batch_u))
                new_y[label, :] = 1
                eps = eps_minibatch_u[label]
                # logpx, logpz, logqz, _gv, _gw = model.dL_dw(v, w, {'x':x_minibatch['x'],'y':new_y}, eps)
                L_unweighted, L_weighted, _gv, _gw = model.dL_weighted_dw(v, w, {'x': x_minibatch_u['x'], 'y': new_y},
                                                                          eps, py[label:label + 1, :])
                _L[label:label + 1, :] = L_unweighted
                for i in v: gv_unlabeled[i] += _gv[i]
                for i in w: gw_unlabeled[i] += _gw[i]
        else:
            # New, should be more efficient. (But is not in practice)
            _y = np.zeros((n_y, n_batch_u * n_y))
            for label in range(n_y):
                _y[label, label * n_batch_u:(label + 1) * n_batch_u] = 1
            _x = np.tile(x_minibatch_u['x'].astype(np.float32), (1, n_y))
            eps = model.gen_eps(n_batch_u * n_y)
            L_unweighted, L_weighted, gv_unlabeled, gw_unlabeled = model.dL_weighted_dw(v, w, {'x': _x, 'y': _y}, eps,
                                                                                        py.reshape((1, -1)))
            _L = L_unweighted.reshape((n_y, n_batch_u))

        r = f_du(*([x_minibatch_u['x']] + list(u.values()) + [np.zeros((1, n_batch_u)), _L]))
        L_unlabeled = r[0]
        gu_unlabeled = dict(zip(u.keys(), r[1:]))

        # Get gradient of prior
        logpu, gu_prior = model_qy.dlogpw_dw(u)
        logpv, logpw, gv_prior, gw_prior = model.dlogpw_dw(v, w)

        # Combine gradients and objective
        gu = {i: ((gu_labeled[i] + gu_unlabeled[i]) * n_minibatches + gu_prior[i]) / (-n_tot) for i in u}
        gv = {i: ((gv_labeled[i] + gv_unlabeled[i]) * n_minibatches + gv_prior[i]) / (-n_tot) for i in v}
        gw = {i: ((gw_labeled[i] + gw_unlabeled[i]) * n_minibatches + gw_prior[i]) / (-n_tot) for i in w}
        f = ((L_labeled.sum() + L_unlabeled.sum()) * n_minibatches + logpu + logpv + logpw) / (-n_tot)

        L[0] += ((L_labeled.sum() + L_unlabeled.sum()) * n_minibatches + logpu + logpv + logpw) / (-n_tot)
        n_L[0] += 1

        # ndict.pNorm(gu_unlabeled)

        return f, {'u': gu, 'v': gv, 'w': gw}

    w_init = {'u': u_init, 'v': v_init, 'w': w_init}

    optimizer = AdaM(f_df, w_init, minibatches, alpha=3e-4, beta1=0.9, beta2=0.999)

    for i in range(n_passes):
        w = optimizer.optimize(num_passes=1)
        LB = L[0] / (1. * n_L[0])
        testset_error = hook(i, w['u'], w['v'], w['w'], LB)
        L[0] = 0
        n_L[0] = 0

    return testset_error

def load_dataset(n_labeled, n_unlabeled, n_classes):
    """
    Loads the data with at least n_labeled labeled samples.

    :param n_labeled: Number of samples with label.
    :return: train_x_labeled, train_y_labeled, train_x_unlabeled, train_y_unlabeled, valid_x, valid_y, test_x, test_y
    """

    # Load dataset
    from hyperspectralData import HyperspectralData

    print("---------------")
    print("Loading labeled samples.")
    x_l, y_l, valid_x, valid_y, test_x, test_y = HyperspectralData().get_labeled_numpy(n_labeled, n_labeled, n_labeled)
    print("Loading unlabeled samples.")
    x_u, y_u, _, _, _, _ = HyperspectralData().get_unlabeled_numpy(n_unlabeled, 1, 1)

    # To one hot encoding:
    y_l = HyperspectralData().to_one_hot(y_l, n_classes)
    y_u = HyperspectralData().to_one_hot(y_u, n_classes)
    valid_y = HyperspectralData().to_one_hot(valid_y, n_classes)
    test_y = HyperspectralData().to_one_hot(test_y, n_classes)

    return x_l, y_l, x_u, y_u, valid_x, valid_y, test_x, test_y


def write_headers(logdir):
    # Write the headers for the csv file output:
    with open(logdir + 'AA_results.txt', 'w') as file:
        # Like a csv file:
        file.write(
            "Step" + ',' + "Time_elapsed" + ',' + "Validset_error" + ',' + "Testset_error" + '\n')
        file.close()
