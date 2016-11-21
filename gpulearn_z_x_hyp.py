import sys
sys.path.append('..')
sys.path.append('../../data/')

import os, numpy as np
import time

import anglepy as ap
import anglepy.paramgraphics as paramgraphics
import anglepy.ndict as ndict

import theano
import theano.tensor as T
from collections import OrderedDict

import preprocessing as pp

def main(n_z, n_hidden, dataset, seed, comment, gfx=True):

    # Initialize logdir
    #---------------------
    # Setasouto:
    # Create the directory to save the outputs files and log.
    #---------------------
    import time
    logdir = 'results/gpulearn_z_x_'+dataset+'_'+str(n_z)+'-'+str(n_hidden)+'_'+comment+'_'+str(int(time.time()))+'/'
    if not os.path.exists(logdir): os.makedirs(logdir)
    print('Logdir:', logdir)

    np.random.seed(seed)

    gfx_freq = 1

    weight_decay = 0
    f_enc, f_dec = lambda x:x, lambda x:x

    # Init data
    if dataset == 'hyper':
        # Hyperspectral images:

        # Import 1 file of the dataset
        # TODO: import more files: Edit hyperspectralData.py

        #I added the hyperspectralData file in the anglepy library
        from hyperspectralData import HyperspectralData
        nsamples = 100000
        train_x, train_y, valid_x, valid_y, test_x, test_y = HyperspectralData().load_numpy(nsamples)

        #Dim input: How it has to be written like an image. We said that is:
        dim_input = (67,4)
        n_x = train_x.shape[0] #Dimension of our data vector.

        x = {'x': train_x.astype(np.float32)}
        x_valid = {'x': valid_x.astype(np.float32)}
        x_test = {'x': test_x.astype(np.float32)}
        # Hyperparameters:
        L_valid = 1
        type_qz = 'gaussianmarg'
        type_pz = 'gaussianmarg'
        nonlinear = 'softplus'
        type_px = 'bernoulli'
        n_train = train_x.shape[1]
        n_batch = 1000
        colorImg = False
        bernoulli_x = False
        byteToFloat = False
        weight_decay = float(n_batch)/n_train
        #Write the hyperparameters used:
        with open(logdir+'AA_hyperparameters.txt', 'w') as file:
            file.write("L_valid: " + str(L_valid) + '\n')
            file.write("type_qz: " + type_qz + '\n')
            file.write("type_pz: " + type_pz + '\n')
            file.write("Nonlinear: " + nonlinear + '\n')
            file.write("type_px: " + type_px + '\n')
            file.write("n_train: " + str(n_train) + '\n')
            file.write("n_batch: " + str(n_batch) + '\n')
            file.write("colorImg: " + str(colorImg) + '\n')
            file.write("bernoulli_x: " + str(bernoulli_x) + '\n')
            file.write("byteToFloat: " + str(byteToFloat) + '\n')
            file.close()
        # Write the headers for the csv file output:
        with open(logdir+'AA_results.txt', 'w') as file:
            # Like a csv file:
            file.write("Step" + ',' + "TimeElapsed" + ',' + "LowerboundMinibatch" + ',' + "LowerboundValid" + ','+ "NumStepNotImproving" + '\n')
            file.close()


    # Construct model
    from anglepy.models import GPUVAE_Z_X
    updates = get_adam_optimizer(learning_rate=3e-4, weight_decay=weight_decay)
    model = GPUVAE_Z_X(updates, n_x, n_hidden, n_z, n_hidden[::-1], nonlinear, nonlinear, type_px, type_qz=type_qz, type_pz=type_pz, prior_sd=100, init_sd=1e-3)

    # Some statistics for optimization
    ll_valid_stats = [-1e99, 0]

    # Progress hook
    def hook(epoch, t, ll):
        '''
        Documented by SetaSouto, may contains errors.

        :epoch: Number of the current step.
        :t: Time elapsed from the beginning.
        :ll: Loglikelihood.
        '''

        if epoch%10 != 0: return

        ll_valid, _ = model.est_loglik(x_valid, n_samples=L_valid, n_batch=n_batch, byteToFloat=byteToFloat)

        # Saves the value of our actual net.
        ndict.savez(ndict.get_value(model.v), logdir+'v')
        ndict.savez(ndict.get_value(model.w), logdir+'w')

        # If the actual ll of the validset is the best:
        if ll_valid > ll_valid_stats[0]:
            ll_valid_stats[0] = ll_valid
            # Reset the numbers of iterations without improving:
            ll_valid_stats[1] = 0
            ndict.savez(ndict.get_value(model.v), logdir+'v_best')
            ndict.savez(ndict.get_value(model.w), logdir+'w_best')
        else:
            ll_valid_stats[1] += 1
            # Stop when not improving validation set performance in 100 iterations
            if ll_valid_stats[1] > 100:
                print("Finished")
                with open(logdir+'hook.txt', 'a') as f:
                    print(f, "Finished")
                exit()

        # This will be showing the current results and write them in a file:
        with open(logdir+'AA_results.txt', 'a') as file:
            # Like a csv file:
            file.write(str(epoch) + ',' + str(t) + ',' + str(ll) + ',' + str(ll_valid) + ',' + str(ll_valid_stats[1]) + '\n')
            file.close()
        print("-------------------------")
        print("Current results:")
        print(" ")
        print("Step:", epoch)
        print("Time elapsed:", t)
        print("Loglikelihood minibatch:", ll)
        print("Loglikelihood validSet:", ll_valid)
        print("N not improving:", ll_valid_stats[1])

        # Graphics: Generate images from the math
        if gfx and epoch%gfx_freq == 0:

            #tail = '.png'
            tail = '-'+str(epoch)+'.png'

            v = {i: model.v[i].get_value() for i in model.v}
            w = {i: model.w[i].get_value() for i in model.w}

            if 'pca' not in dataset and 'random' not in dataset and 'normalized' not in dataset:

                if 'w0' in v:
                    image = paramgraphics.mat_to_img(f_dec(v['w0'][:].T), dim_input, True, colorImg=colorImg)
                    image.save(logdir+'q_w0'+tail, 'PNG')

                image = paramgraphics.mat_to_img(f_dec(w['out_w'][:]), dim_input, True, colorImg=colorImg)
                image.save(logdir+'out_w'+tail, 'PNG')

                if 'out_unif' in w:
                    image = paramgraphics.mat_to_img(f_dec(w['out_unif'].reshape((-1,1))), dim_input, True, colorImg=colorImg)
                    image.save(logdir+'out_unif'+tail, 'PNG')

                if n_z == 2:
                    n_width = 10
                    import scipy.stats
                    z = {'z':np.zeros((2,n_width**2))}
                    for i in range(0,n_width):
                        for j in range(0,n_width):
                            z['z'][0,n_width*i+j] = scipy.stats.norm.ppf(float(i)/n_width+0.5/n_width)
                            z['z'][1,n_width*i+j] = scipy.stats.norm.ppf(float(j)/n_width+0.5/n_width)

                    x, _, _z = model.gen_xz({}, z, n_width**2)
                    if dataset == 'mnist':
                        x = 1 - _z['x']
                    image = paramgraphics.mat_to_img(f_dec(_z['x']), dim_input)
                    image.save(logdir+'2dmanifold'+tail, 'PNG')
                else:
                    _x, _, _z_confab = model.gen_xz({}, {}, n_batch=144)
                    x_samples = _z_confab['x']
                    image = paramgraphics.mat_to_img(f_dec(x_samples), dim_input, colorImg=colorImg)
                    image.save(logdir+'samples'+tail, 'PNG')

                    #x_samples = _x['x']
                    #image = paramgraphics.mat_to_img(x_samples, dim_input, colorImg=colorImg)
                    #image.save(logdir+'samples2'+tail, 'PNG')

            else:
                # Model with preprocessing

                if 'w0' in v:
                    image = paramgraphics.mat_to_img(f_dec(v['w0'][:].T), dim_input, True, colorImg=colorImg)
                    image.save(logdir+'q_w0'+tail, 'PNG')

                image = paramgraphics.mat_to_img(f_dec(w['out_w'][:]), dim_input, True, colorImg=colorImg)
                image.save(logdir+'out_w'+tail, 'PNG')

                _x, _, _z_confab = model.gen_xz({}, {}, n_batch=144)
                x_samples = f_dec(_z_confab['x'])
                x_samples = np.minimum(np.maximum(x_samples, 0), 1)
                image = paramgraphics.mat_to_img(x_samples, dim_input, colorImg=colorImg)
                image.save(logdir+'samples'+tail, 'PNG')



    # Optimize
    #SFO
    dostep = epoch_vae_adam(model, x, n_batch=n_batch, bernoulli_x=bernoulli_x, byteToFloat=byteToFloat)
    loop_va(dostep, hook)

    pass

# Training loop for variational autoencoder
def loop_va(doEpoch, hook, n_epochs=9999999):

    t0 = time.time()
    for t in range(1, n_epochs):
        L = doEpoch()
        hook(t, time.time() - t0, L)

    print('Optimization loop finished')

# Learning step for variational auto-encoder
def epoch_vae_adam(model, x, n_batch=100, convertImgs=False, bernoulli_x=False, byteToFloat=False):
    print('Variational Auto-Encoder', n_batch)

    def doEpoch():

        from collections import OrderedDict

        n_tot = next(iter(x.values())).shape[1]
        idx_from = 0
        L = 0
        while idx_from < n_tot:
            idx_to = min(n_tot, idx_from+n_batch)
            x_minibatch = ndict.getCols(x, idx_from, idx_to)
            idx_from += n_batch
            if byteToFloat: x_minibatch['x'] = x_minibatch['x'].astype(np.float32)/256.
            if bernoulli_x: x_minibatch['x'] = np.random.binomial(n=1, p=x_minibatch['x']).astype(np.float32)

            # Do gradient ascent step
            L += model.evalAndUpdate(x_minibatch, {}).sum()
            #model.profmode.print_summary()

        L /= n_tot

        return L

    return doEpoch


def get_adam_optimizer(learning_rate=0.001, decay1=0.1, decay2=0.001, weight_decay=0.0):
    #print('AdaM', learning_rate, decay1, decay2, weight_decay)
    def shared32(x, name=None, borrow=False):
        return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)

    def get_optimizer(w, g):
        updates = OrderedDict()

        it = shared32(0.)
        updates[it] = it + 1.

        fix1 = 1.-(1.-decay1)**(it+1.) # To make estimates unbiased
        fix2 = 1.-(1.-decay2)**(it+1.) # To make estimates unbiased
        lr_t = learning_rate * T.sqrt(fix2) / fix1

        for i in w:

            gi = g[i]
            if weight_decay > 0:
                gi -= weight_decay * w[i] #T.tanh(w[i])

            # mean_squared_grad := E[g^2]_{t-1}
            mom1 = shared32(w[i].get_value() * 0.)
            mom2 = shared32(w[i].get_value() * 0.)

            # Update moments
            mom1_new = mom1 + decay1 * (gi - mom1)
            mom2_new = mom2 + decay2 * (T.sqr(gi) - mom2)

            # Compute the effective gradient and effective learning rate
            effgrad = mom1_new / (T.sqrt(mom2_new) + 1e-10)

            effstep_new = lr_t * effgrad

            # Do update
            w_new = w[i] + effstep_new

            # Apply update
            updates[w[i]] = w_new
            updates[mom1] = mom1_new
            updates[mom2] = mom2_new

        return updates

    return get_optimizer
