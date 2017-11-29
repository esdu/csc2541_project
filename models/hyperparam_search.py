import time

import numpy as np
import tensorflow as tf
from nnmf_svi_eddie import save_graph_parameters
from nnmf_svi_eddie import NNMF

VERSION = 3

def get_hypers_config(seed, version=VERSION):
    import random
    random.seed(seed)

    if version == 2:
        D = random.randint(5,100)
        Dp = random.randint(5,100)

        nn_hidden_layer_dims = []
        n_layers = random.randint(1,5)
        for l in range(n_layers):
            nn_hidden_layer_dims.append(random.randint(5,100))

        batch_size = random.randint(100,400)
        n_samples = random.choice([1,10,100])

        pZ_prior_stddev = random.randrange(200) / 100 # 0 to 2
        pR_stddev = random.randrange(200) / 100 # 0 to 2

        nn_W_init_mean = 0.
        nn_W_init_stddev = random.randrange(200) / 100 # 0 to 2
        nn_b_init_mean = 0.
        nn_b_init_stddev = random.randrange(200) / 100 # 0 to 2

        optimizer = 'adam'
        lr_init = random.choice([0.01,0.1,1.0])
        lr_decay_steps = random.choice([100,200,300])
        lr_decay_rate = random.choice([0.9,0.95,0.99])

        return {
            'D': D,
            'Dp': Dp,
            'nn_hidden_layer_dims': nn_hidden_layer_dims,
            'batch_size': batch_size,
            'n_samples': n_samples,
            'pZ_prior_stddev': pZ_prior_stddev,
            'pR_stddev': pR_stddev,
            'nn_W_init_mean': nn_W_init_mean,
            'nn_W_init_stddev': nn_W_init_stddev,
            'nn_b_init_mean': nn_b_init_mean,
            'nn_b_init_stddev': nn_b_init_stddev,
            'optimizer': optimizer,
            'lr_init': lr_init,
            'lr_decay_steps': lr_decay_steps,
            'lr_decay_rate': lr_decay_rate
        }
    elif version == 3:
        D  = random.choice([10,35,60])
        Dp = random.choice([10,35,60])

        nn_hidden_layer_dims = []
        n_layers = random.randint(2,3)
        for l in range(n_layers):
            nn_hidden_layer_dims.append(50)#random.randint(5,100))

        batch_size = 200 #random.randint(100,400)
        n_samples = 50 #random.choice([1,10,100])

        pZ_prior_stddev = random.choice([0.5,1,1.5]) #random.randrange(200) / 100 # 0 to 2
        pR_stddev = 1 #random.randrange(200) / 100 # 0 to 2

        nn_W_init_mean = 0.
        nn_W_init_stddev = (random.randrange(75) + 25) / 100 # 0 to 2
        nn_b_init_mean = 0.
        nn_b_init_stddev = (random.randrange(50) + 75) / 100 # 0 to 2

        optimizer = 'adam'
        lr_init = random.choice([0.005,0.01,0.1])
        lr_decay_steps = 100 #random.choice([100,200,300])
        lr_decay_rate = 0.9 #random.choice([0.9,0.95,0.99])

        return {
            'D': D,
            'Dp': Dp,
            'nn_hidden_layer_dims': nn_hidden_layer_dims,
            'batch_size': batch_size,
            'n_samples': n_samples,
            'pZ_prior_stddev': pZ_prior_stddev,
            'pR_stddev': pR_stddev,
            'nn_W_init_mean': nn_W_init_mean,
            'nn_W_init_stddev': nn_W_init_stddev,
            'nn_b_init_mean': nn_b_init_mean,
            'nn_b_init_stddev': nn_b_init_stddev,
            'optimizer': optimizer,
            'lr_init': lr_init,
            'lr_decay_steps': lr_decay_steps,
            'lr_decay_rate': lr_decay_rate
        }

def save_output_csv(fname, data_list):
    import csv
    with open(fname, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_list)

_SESS = None
def get_new_session():
    global _SESS
    if _SESS is not None:
        _SESS.close()
    _SESS = tf.Session()
    return _SESS

def hypersearch(folder, model_name, dataset_name, seed, n_iter, R, train_mask, valid_mask, verbose=False):
    import os
    os.makedirs(folder, exist_ok=True)

    csv_output = "{}/{}_{}_{}_{}.csv".format(folder, model_name, dataset_name, seed, n_iter)
    mdl_output = "{}/{}_{}_{}_{}.pkl".format(folder, model_name, dataset_name, seed, n_iter)

    if os.path.exists(csv_output):
        print("Skip #{}".format(seed))
    else:
        hypers = get_hypers_config(seed)
        if verbose: print(hypers)

        tf.reset_default_graph()

        sess = get_new_session()
        with sess.as_default():
            model = NNMF(ratings_matrix=R, **hypers)
            losses = model.train(train_mask, n_iter=n_iter, verbose=verbose)
            #if verbose:
            #    plt.plot(losses)
            #    plt.show()

            # Evaluation
            if verbose: print('Evaluating...')
            def get_mse(mask):
                _start_time = time.time()
                # We evaluate 10 draws at a time so everything fits into memory.
                results_batches = []
                for _ in range(1): # 100 draws total
                    idx_i_all, idx_j_all = np.where(mask)
                    feed_dict = {
                        model.test_idx_i: idx_i_all,
                        model.test_idx_j: idx_j_all,
                        model.n_test_samples: 100
                    }
                    results_batch = np.squeeze(sess.run(model.sample_rhats, feed_dict))
                    results_batches.append(np.mean(results_batch, axis=0))
                results_batches = np.array(results_batches)
                results = np.mean(results_batches, axis=0)
                mse = np.mean(np.square(results - R[idx_i_all, idx_j_all]))
                if verbose: print('Evaluation time: {}s'.format(time.time() - _start_time))
                return mse

            #train_mse = get_mse(train_mask)
            valid_mse = get_mse(valid_mask)

            # Output
            # XXX We'll only output validation set
            #data_list = [model_name, dataset_name, seed, n_iter, losses[-1], train_mse, valid_mse]
            data_list = [model_name, dataset_name, seed, n_iter, losses[-1], valid_mse]

            if verbose: print(data_list)
            save_output_csv(csv_output, data_list)
            save_graph_parameters(mdl_output)
            if verbose: print("Done #{}".format(seed))

def _get_training_matrix():
    """Adapted from Soon's code"""
    import sys
    sys.path.append('..')

    import copy
    from sclrecommender.mask import MaskGenerator
    from sclrecommender.mask import RandomMaskGenerator
    # TODO: from sclrecommender.mask import TimeMaskGenerator
    # TODO: from sclrecommender.mask import ColdUserMaskGenerator
    # TODO: from sclrecommender.mask import ColdItemMaskGenerator
    from sclrecommender.mask import LegalMoveMaskGenerator
    from sclrecommender.matrix import RatingMatrix
    from sclrecommender.matrix import PositiveNegativeMatrix
    from sclrecommender.parser import ExampleParser
    from sclrecommender.parser import MovieLensParser

    # Soon's code provides us with training matrix
    from sclrecommender.parser import MovieLensParser
    seedNum = 196
    np.random.seed(seedNum)
    dataDirectory = "../ml-100k"
    mlp = MovieLensParser(dataDirectory)
    ratingMatrix = mlp.getRatingMatrixCopy()

    rmTruth = RatingMatrix(ratingMatrix)

    trainSplit = 0.8
    randomMaskTrain, randomMaskTest = RandomMaskGenerator(rmTruth.getRatingMatrix(), trainSplit).getMasksCopy()

    rmTrain = copy.deepcopy(rmTruth)
    rmTrain.applyMask(randomMaskTrain)

    trainMatrix = rmTrain.getRatingMatrix()

    return trainMatrix

def get_training_matrix():
    import os
    import _pickle

    if not os.path.exists('_rating_matrix.pkl'):
        _pickle.dump(_get_training_matrix(), open('_rating_matrix.pkl', 'wb'))

    return _pickle.load(open('_rating_matrix.pkl', 'rb'))

if __name__ == '__main__':
    #
    # Load data, split into train and validation sets.
    #

    # ---
    import random
    some_really_random_number = random.randint(0,999999999)
    # ---

    R = get_training_matrix()
    mask = R>0

    np.random.seed(1337) # Set a fixed seed, so we get a fixed mask.
    TRAIN_MASK = (np.random.binomial(1, 0.9, size=R.shape)) & mask
    VALID_MASK = (1-TRAIN_MASK) & mask

    np.random.seed(some_really_random_number) # Go back to "true" randomness.

    #
    # Run it :)
    #

    #for _ in range(100):
    while True:
        MODEL_NAME = 'NNMF'
        DATASET_NAME = 'TEST'
        SEED = np.random.randint(999999999)
        if VERSION == 2:
            N_ITER = 200
        elif VERSION == 3:
            N_ITER = 500
        OUTPUT_FOLDER = 'hypersearch_v{}'.format(VERSION)

        print('Seed: {}'.format(SEED))
        hypersearch(OUTPUT_FOLDER, MODEL_NAME, DATASET_NAME, SEED, N_ITER, R, TRAIN_MASK, VALID_MASK, verbose=True)
