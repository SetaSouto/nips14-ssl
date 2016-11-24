import learn_yz_x_hyp

n_passes = 3000
n_hidden = (500,)
seed = 0
alpha = 0.1
n_minibatches = 50
n_labeled = 100
n_unlabeled = 100
n_classes = 100
learn_yz_x_hyp.main(n_passes, n_hidden, seed, alpha, n_minibatches, n_labeled, n_unlabeled, n_classes)