import numpy as np
import tensorflow as tf
import edward as ed
try:
    from nnmf_svi_eddie import create_optimizer
except Exception:
    from models.nnmf_svi_eddie import create_optimizer

class SimpleMatrixFactorization:
    def __init__(self, ratings_matrix, hidden_dim=30,
                 batch_size=200, n_samples=1, pR_stddev=1.,
                 lr_init=0.1, lr_decay_steps=200, lr_decay_rate=0.99,
                 BERN=False):
        """
        Computes R = UV' with SVI.

        :param ratings_matrix: The full ratings matrix.
            Ratings should be positive, and a rating of 0 means unknown rating.
        :param hidden_dim: Hidden dim size for for U and V.
        :param batch_size: For each itration of SVI, how many samples from
            ratings matrix to use.
            Bigger batch => more stable gradients.
        :param n_sample: For each iteration of SVI, how many latent samples to
            draw to estimate the gradient.
            Higher n_sample => more stable gradients.
        :param pR_stddev: The model's stddev, ie. for p(r_{ij}|U,V).
        """
        # TODO: How to update this with more data? Right now it needs to retrain from beginning.
        #       Re-building a graph every time isn't ideal. We need to at least clean up the old graph.
        #       Workaround: user call tf.reset_default_graph before creating this.

        self.R_ = ratings_matrix
        N, M = self.R_.shape; self.N = N; self.M = M
        D = hidden_dim
        self.batch_size = batch_size

        # We use r_ph to feed in only the elements in R_ that idx_i and idx_j correspond to.
        self.r_ph  = tf.placeholder(tf.float32, name="batch_r")
        self.idx_i = tf.placeholder(tf.int32, name="idx_i")
        self.idx_j = tf.placeholder(tf.int32, name="idx_j")

        # "Priors" p(Z)
        self.U = ed.models.Normal(loc=tf.zeros([N, D]), scale=tf.ones([N, D]))
        self.V = ed.models.Normal(loc=tf.zeros([M, D]), scale=tf.ones([M, D]))

        # P(X|Z)
        U_selected = tf.gather(self.U, self.idx_i)
        V_selected = tf.gather(self.V, self.idx_j)
        means = tf.reduce_sum(tf.multiply(U_selected, V_selected), axis=1)
        if BERN:
            self.R = ed.models.TransformedDistribution(
              distribution=ed.models.Bernoulli(logits=means, dtype=tf.float32),#ds.Normal(loc=0., scale=1.),
              bijector=tf.contrib.distributions.bijectors.Affine(shift=1.))
        else:
            self.R = ed.models.Normal(loc=means, scale=pR_stddev*tf.ones(self.batch_size))

        # VI
        self.qU = ed.models.Normal(loc=tf.Variable(tf.zeros([N, D]), name="qU_mean"),
                                   scale=tf.Variable(tf.ones([N, D]), name="qU_var"))
        self.qV = ed.models.Normal(loc=tf.Variable(tf.zeros([M, D]), name="qV_mean"),
                                   scale=tf.Variable(tf.ones([M, D]), name="qV_var"))

        # Inference
        self.inference = ed.KLqp({self.U: self.qU, self.V: self.qV}, data={self.R: self.r_ph})
        _optimizer, global_step = create_optimizer('adam', lr_init=lr_init, lr_decay_steps=lr_decay_steps, lr_decay_rate=lr_decay_rate)
        self.inference.initialize(scale={self.R: N*M/self.batch_size}, n_samples=n_samples,
                                  optimizer=_optimizer, global_step=global_step)

        # Note: global_variables_initializer has to be run after creating inference.
        ed.get_session().run(tf.global_variables_initializer())

        # Testing
        self.test_idx_i = tf.placeholder(tf.int32, name="test_idx_i")
        self.test_idx_j = tf.placeholder(tf.int32, name="test_idx_j")
        self.n_test_samples = tf.placeholder(tf.int32, name="n_test_samples")

        self.qU_samples = tf.gather(self.qU.sample(self.n_test_samples), self.test_idx_i, axis=1)
        self.qV_samples = tf.gather(self.qV.sample(self.n_test_samples), self.test_idx_j, axis=1)
        if BERN:
            self.sample_rhats = tf.sigmoid(tf.reduce_sum(tf.multiply(self.qU_samples, self.qV_samples), axis=-1)) + 1
        else:
            self.sample_rhats = tf.reduce_sum(tf.multiply(self.qU_samples, self.qV_samples), axis=-1)

        # for Validation
        self.r_map_estimates = tf.reduce_mean(tf.squeeze(self.sample_rhats), axis=0)
        self.mse = tf.reduce_mean(tf.square(self.r_ph - self.r_map_estimates))
        self.loss = self.inference.loss

        # Misc

        self.saver = tf.train.Saver(max_to_keep=0)


    def sample_user_ratings(self, user_index, n_samples=100):
        idx_i = [user_index] * self.M
        idx_j = list(range(self.M))
        feed_dict = {
            self.test_idx_i: idx_i,
            self.test_idx_j: idx_j,
            self.n_test_samples: n_samples
        }
        return np.squeeze(ed.get_session().run(self.sample_rhats, feed_dict))


    #def mse(self):
    #    R_samples_ = self.rhat_samples()
    #    return np.mean(np.square(np.mean(R_samples_, axis=0) - self.R_)[np.where(self.mask)])


    #def train(self, mask, n_iter=1000, verbose=False):
    #    """
    #    :param mask: Same size as ratings_matrix.
    #        A 0 indicates to not use this rating, else use this rating.
    #        If None given, ratings_matrix will be used as the mask.
    #    :param n_iter: How many iterations of SVI to run.
    #    """
    #    seen_indices = np.array(np.where(mask))
    #    info_dicts = []
    #    for _ in range(n_iter):
    #        # Train on a batch of BATCH_SIZE random elements each iteration.
    #        rand_idx = np.random.choice(seen_indices.shape[1], self.batch_size, replace=False)
    #        idx_i_ = seen_indices[0, rand_idx]
    #        idx_j_ = seen_indices[1, rand_idx]
    #        feed_dict = {
    #            self.idx_i: idx_i_,
    #            self.idx_j: idx_j_,
    #            self.r_ph: self.R_[idx_i_, idx_j_]
    #        }
    #        info_dict = self.inference.update(feed_dict=feed_dict)
    #        info_dicts.append(info_dict)

    #        # TODO print out progress without using edward
    #        #if verbose: self.inference.print_progress(info_dict)

    #    losses = [x['loss'] for x in info_dicts]
    #    return losses

    def train(self, mask, valid_mask=None, n_iter=1000, verbose=False):
        """
        :param mask: Same size as ratings_matrix.
            A 0 indicates to not use this rating, else use this rating.
            If None given, ratings_matrix will be used as the mask.
        :param valid_mask: Same size as ratings_matrix.
            Used to access when the model has trained to convergence.
        :param n_iter: How many iterations of SVI to run.
        """
        info_dicts = []
        seen_indices = np.array(np.where(mask))

        if valid_mask is not None:
            valid_losses = []
            valid_indices = np.array(np.where(valid_mask))

            valid_mses = []
            valid_mse_idx_i = valid_indices[0, :]
            valid_mse_idx_j = valid_indices[1, :]
            valid_mse_feed_dict = {
                self.test_idx_i: valid_mse_idx_i,
                self.test_idx_j: valid_mse_idx_j,
                self.n_test_samples: 20,
                self.r_ph: self.R_[valid_mse_idx_i, valid_mse_idx_j]
            }

        for _ in range(n_iter):
            # TODO is there a problem training with random batches? Look into lit more.
            # Train on a batch of self.batch_size random elements each iteration.
            rand_idx = np.random.choice(seen_indices.shape[1], self.batch_size, replace=False)
            idx_i = seen_indices[0, rand_idx]
            idx_j = seen_indices[1, rand_idx]
            feed_dict = {
                self.idx_i: idx_i,
                self.idx_j: idx_j,
                self.r_ph: self.R_[idx_i, idx_j]
            }
            info_dict = self.inference.update(feed_dict=feed_dict)
            info_dicts.append(info_dict)

            # TODO print out progress without using edward
            if verbose: self.inference.print_progress(info_dict)

            if valid_mask is not None:
                # Validation
                # TODO either we use self.mse
                # Or we need to make the batch size bigger
                # Or we need to make the run this multiple times
                valid_idx = np.random.choice(valid_indices.shape[1], self.batch_size, replace=False)

                valid_idx_i = valid_indices[0, valid_idx]
                valid_idx_j = valid_indices[1, valid_idx]
                valid_feed_dict = {
                    self.idx_i: valid_idx_i,
                    self.idx_j: valid_idx_j,
                    self.r_ph: self.R_[valid_idx_i, valid_idx_j]
                }

                valid_loss = ed.get_session().run(self.loss, valid_feed_dict)
                valid_losses.append(valid_loss)

                valid_mse = ed.get_session().run(self.mse, valid_mse_feed_dict)
                valid_mses.append(valid_mse)

        losses = [x['loss'] for x in info_dicts]

        if valid_mask is not None:
            return losses, valid_losses, valid_mses
        else:
            return losses
