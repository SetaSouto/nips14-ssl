import learn_yz_x_hyp

n_passes = 3000
n_hidden = (500,)
seed = 0
alpha = 0.1
n_minibatches = 100
learn_yz_x_hyp.main(n_passes, n_hidden, seed, alpha, n_minibatches)