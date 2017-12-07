from models.nnmf_svi_eddie import NNMF as _NNMF
from models.nnmf_svi_eddie import save_graph_parameters, load_graph_parameters
from uncertaintyModel import UncertaintyModel
import tensorflow as tf
import edward as ed

class NNMF(UncertaintyModel):
    """
    Thin wrapper around our model to conform to Soon's API.
    It also allows us to manage session/graph/hyperparams if we want.
    """

    def __init__(self, ratingMatrix, seed=None):
        self.ratingMatrix = ratingMatrix
        self.sess = None
        self.reset(seed=seed)

    def reset(self, seed=None):
        tf.reset_default_graph()
        if seed is not None:
            ed.set_seed(seed) # sets seed for both tf and numpy

        if self.sess is not None:
            self.sess.close()
        self.sess = tf.Session()

        with self.sess.as_default():
            #self.model = _NNMF(
            #    self.ratingMatrix,
            #    D=10, Dp=60, pZ_prior_stddev=1., pR_stddev=1.,
            #    nn_hidden_layer_dims=[50,50,50], nn_W_init_mean=0., nn_W_init_stddev=1., nn_b_init_mean=0., nn_b_init_stddev=1.,
            #    batch_size=200, n_samples=10,
            #    optimizer='adam', lr_init=0.1, lr_decay_steps=100, lr_decay_rate=0.9
            #)
            self.model = _NNMF(
                self.ratingMatrix,
                D=60, Dp=35, pZ_prior_stddev=0.5, pR_stddev=1.,
                nn_hidden_layer_dims=[50,50,50], nn_W_init_mean=0., nn_W_init_stddev=0.51, nn_b_init_mean=0., nn_b_init_stddev=1.18,
                batch_size=200, n_samples=50,
                optimizer='adam', lr_init=0.01, lr_decay_steps=100, lr_decay_rate=0.9
            )

    def save(self, fname):
        with self.sess.as_default():
            save_graph_parameters(fname)
        return fname

    def load(self, fname):
        with self.sess.as_default():
            load_graph_parameters(fname)

    def train(self, legalTrainIndices, n_iter=2000):
        with self.sess.as_default():
            losses = self.model.train(mask=legalTrainIndices, n_iter=n_iter)
        return losses

    def sample_for_user(self, user_index, num_samples):
        # return (k, m) matrix of k samples for user i
        with self.sess.as_default():
            return self.model.sample_user_ratings(user_index, num_samples)
