from models.simple import SimpleMatrixFactorization as _PMF
from models.nnmf_svi_eddie import save_graph_parameters, load_graph_parameters
from uncertaintyModel import UncertaintyModel
import tensorflow as tf
import edward as ed

class PMF(UncertaintyModel):
    """
    Thin wrapper around our model to conform to Soon's API.
    It also allows us to manage session/graph/hyperparams if we want.
    """

    def __init__(self, ratingMatrix):
        self.ratingMatrix = ratingMatrix
        self.sess = None
        self.reset()

    def reset(self, seed=None):
        tf.reset_default_graph()
        ed.set_seed(seed) # sets seed for both tf and numpy

        if self.sess is not None:
            self.sess.close()
        self.sess = tf.Session()

        with self.sess.as_default():
            self.model = _PMF(
                self.ratingMatrix,
                hidden_dim=70, # To match NNMF 60 + 10 hidden dims.
                batch_size=200, n_samples=10, pR_stddev=1.)

    def save(self, fname):
        with self.sess.as_default():
            save_graph_parameters(fname)
        return fname

    def load(self, fname):
        with self.sess.as_default():
            load_graph_parameters(fname)

    def train(self, legalTrainIndices, n_iter=1000):
        with self.sess.as_default():
            losses = self.model.train(mask=legalTrainIndices, n_iter=n_iter)
        return losses

    def sample_for_user(self, user_index, num_samples):
        # return (k, m) matrix of k samples for user i
        with self.sess.as_default():
            return self.model.sample_user_ratings(user_index, num_samples)
