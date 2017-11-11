import numpy as np

def make_bandits_mask(mask, R):
    '''
    Return a new mask where if R[i,j] == 0, then mask[i,j] = 1

    eg.
    R     =  4 3 2
             0 3 3

    mask  =  1 1 1
             0 1 0

    mask' =  1 1 1
             1 1 0

    Effectively, where mask'[i,j] == 0 is what we can reveal to bandits.
    '''
    m = np.copy(mask)
    m[np.where(R == 0)] = 1
    return m
