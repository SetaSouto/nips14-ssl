import gpulearn_z_x
import sys

if 'svhn' in sys.argv[1]:
    gpulearn_z_x.main(dataset=sys.argv[1], n_z=300, n_hidden=(500,500), seed=0, comment='', gfx=True)
elif sys.argv[1] == 'mnist':
    n_hidden = (500,500)
    if len(sys.argv) == 4:
        n_hidden = [int(sys.argv[2])] * int(sys.argv[3])
    gpulearn_z_x.main(dataset='mnist', n_z=50, n_hidden=n_hidden, seed=0, comment='', gfx=True)
elif sys.argv[1] == 'hyperspectral':
    n_hidden = (500,500)
    gpulearn_z_x.main(dataset='hyper', n_z = 50, n_hidden=n_hidden, seed=0, comment='', gfx=True)
