import numpy as np
from rnncomp.dataman import *

fi = np.load("datasets/dataset_flatcls_0.5_2_3_0.npz")
desc = fi["class_desc"].item()
nengo_dat, nengo_cor = make_run_args_nengo(fi)
print(nengo_dat)
print(nengo_dat.shape)

dim_last = nengo_dat.reshape((3, 510, 2))
print(dim_last[0, :, :])

comb = dim_last.reshape((3*510, 1, 2))
print("Chunk A")
print(comb[:510, 0, :])
print("Chunk B")
print(comb[510:2*510, 0, :])
print("Chunk C")
print(comb[2*510:3*510, 0, :])

t_steps = 10
n_classes = 3
sig_num = 1
pause_size = 5

print(nengo_cor)
print(nengo_cor[0])
print(nengo_cor.shape)

re_cor = np.repeat(nengo_cor, t_steps, axis=0).reshape((-1, 1, n_classes))
print(re_cor[:3])
print(re_cor.shape)

re_cor[1]

tot_sigs = n_classes*sig_num
zer = np.zeros((tot_sigs, pause_size, n_classes), dtype=np.int8)
re_zer = np.repeat(nengo_cor, t_steps, axis=0).reshape((tot_sigs, -1, n_classes))
print(zer.shape)
print(re_zer.shape)

cor = np.concatenate((zer, re_zer), axis=1).reshape(-1, 1, n_classes)
print(cor)
print(cor.shape)

