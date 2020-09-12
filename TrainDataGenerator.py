import numpy as np


def train_data_generator(batch_size, container):
    size = len(container)
    i = 0
    while True:
        if i + batch_size >= size:
            i = 0
        res1, res2 = container[i]
        i += 1
        inp_batch = np.zeros((batch_size, *res1.shape)) 
        outp_batch = np.zeros((batch_size, *res2.shape))
        inp_batch[0, :] = res1
        outp_batch[0, :] = res2
        for k in range(1, batch_size):
            inp_batch[k, :] = container[i][0]
            outp_batch[k, :] = container[i][1]
            i += 1
        yield (inp_batch, outp_batch)
        
        