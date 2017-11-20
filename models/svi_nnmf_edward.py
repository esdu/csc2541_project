import numpy as np
import tensorflow as tf
import edward as ed
from matplotlib import pyplot as plt

# turn off trainable for nn parameters
class NNMFCore:
    def __init__(self, U_selected, V_selected, U_prime_selected, V_prime_selected , D, D_prime, hidden_units_per_layer=50, sampling=None):
        self.U_selected, self.V_selected, self.U_prime_selected, self.V_prime_selected = U_selected, V_selected, U_prime_selected, V_prime_selected
        self.D = D
        self.D_prime = D_prime
        self.nhid = hidden_units_per_layer  
        self.nn_parameters = []
        self.sampling = sampling
        self.build_graph()
            
    def nn_inputs_preprocess(self):
        self.nn_inputs = tf.concat([self.U_selected, self.V_selected, tf.multiply(self.U_prime_selected, self.V_prime_selected)], 1)
        
    def nn_fclayers(self):
        
        with tf.variable_scope('nn_parameters', reuse = self.sampling):
            #fc1
            self.fc1W = tf.get_variable(name='fc1_weights', initializer=tf.random_normal([2*self.D + self.D_prime, self.nhid], dtype=tf.float32, stddev=1e-1))
            self.fc1b = tf.get_variable(name='fc1_bias', initializer=tf.random_normal([self.nhid], dtype=tf.float32, stddev=1e-1))
            fc1_in = tf.nn.bias_add(tf.matmul(self.nn_inputs, self.fc1W), self.fc1b)
            fc1 = tf.nn.sigmoid(fc1_in)
            self.nn_parameters += [self.fc1W, self.fc1b]
            
            #fc2
            self.fc2W = tf.get_variable(name='fc2_weights', initializer=tf.random_normal([self.nhid, self.nhid], dtype=tf.float32, stddev=1e-1))
            self.fc2b = tf.get_variable(name='fc2_bias', initializer=tf.random_normal([self.nhid], dtype=tf.float32, stddev=1e-1))
            fc2_in = tf.nn.bias_add(tf.matmul(fc1, self.fc2W), self.fc2b)
            fc2 = tf.nn.sigmoid(fc2_in)
            self.nn_parameters += [self.fc2W, self.fc2b]
            
            #fc3
            self.fc3W = tf.get_variable(name='fc3_weights', initializer=tf.random_normal([self.nhid, 1], dtype=tf.float32, stddev=1e-1))
            self.fc3b = tf.get_variable(name='fc3_bias', initializer=tf.random_normal([1], dtype=tf.float32, stddev=1e-1))
            fc3_in = tf.nn.bias_add(tf.matmul(fc2, self.fc3W), self.fc3b)
            self.prediction = tf.reshape(fc3_in,shape=[-1])
            self.nn_parameters += [self.fc3W, self.fc3b]
            
    def build_graph(self):
        self.nn_inputs_preprocess()
        self.nn_fclayers()
            
    def load_nn_parameters(self, file, sess):
        trained_nn_parameters = cPickle.load(open(file, "rb"))
        nn_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc_layers")
        keys = sorted(trained_nn_parameters.keys())
        for i in range(len(self.nn_parameters)):
            sess.run(self.nn_parameters[i].assign(trained_nn_parameters[keys[i]]))
            
            
class SVI_NNMF:
    def __init__(self, dataset, D=10, D_prime=10, mapping=NNMFCore):
        # infer num_user and num_items from dataset dimension
        self.dataset = dataset
        self.num_users, self.num_items = self.dataset.shape
        self.D = D
        self.D_prime = D_prime 
        self.build_graph()     
    
    def variational_layer(self):
        self.qU_mu = tf.Variable(tf.random_normal([self.num_users, self.D]))
        self.qU_var = tf.Variable(tf.random_normal([self.num_users, self.D]))
        self.qV_mu = tf.Variable(tf.random_normal([self.num_items, self.D]))
        self.qV_var = tf.Variable(tf.random_normal([self.num_items, self.D]))
        self.qU_prime_mu = tf.Variable(tf.random_normal([self.num_users, self.D_prime]))
        self.qU_prime_var = tf.Variable(tf.random_normal([self.num_users, self.D_prime]))
        self.qV_prime_mu = tf.Variable(tf.random_normal([self.num_items, self.D_prime]))
        self.qV_prime_var = tf.Variable(tf.random_normal([self.num_items, self.D_prime]))
        
        self.qU = ed.models.Normal(loc=self.qU_mu, scale=tf.nn.softplus(self.qU_var))
        self.qV = ed.models.Normal(loc=self.qV_mu, scale=tf.nn.softplus(self.qV_var))
        self.qU_prime = ed.models.Normal(loc=self.qU_prime_mu, scale=tf.nn.softplus(self.qU_prime_var))
        self.qV_prime = ed.models.Normal(loc=self.qV_prime_mu, scale=tf.nn.softplus(self.qV_prime_var))
    
    def latent_factors(self):
        # Prior
        self.U = ed.models.Normal(loc=tf.zeros([self.num_users, self.D]), scale=10.0*tf.ones([self.num_users, self.D]))
        self.V = ed.models.Normal(loc=tf.zeros([self.num_items, self.D]), scale=10.0*tf.ones([self.num_items, self.D]))
        self.U_prime = ed.models.Normal(loc=tf.zeros([self.num_users, self.D_prime]), scale=10.0*tf.ones([self.num_users, self.D_prime]))
        self.V_prime = ed.models.Normal(loc=tf.zeros([self.num_items, self.D_prime]), scale=10.0*tf.ones([self.num_items, self.D_prime]))
        
    def feed_forward(self, decoder=NNMFCore, parameter_file=None):
        self.user_indices = tf.placeholder(tf.int32, shape = [None])
        self.item_indices = tf.placeholder(tf.int32, shape = [None])
      
        self.ratings = tf.placeholder(tf.float32, shape = [None, None])  
        
        self.U_selected = tf.gather(self.U, self.user_indices)
        self.V_selected = tf.gather(self.V, self.item_indices)
        self.U_prime_selected = tf.gather(self.U_prime, self.user_indices)
        self.V_prime_selected = tf.gather(self.V_prime, self.item_indices)       
        
        self.decoder = decoder(self.U_selected, self.V_selected, self.U_prime_selected, self.V_prime_selected, self.D, self.D_prime)
        
    def posterior_inference(self):
        self.R_hats = ed.models.Normal(loc=self.decoder.prediction, scale=0.3*tf.ones(tf.shape(self.decoder.prediction)))
        ratings_indices = tf.concat([tf.expand_dims(self.user_indices, axis=1), tf.expand_dims(self.item_indices, axis=1)],1)
        self.ratings_selected = tf.gather_nd(self.ratings, ratings_indices)
        
        # inference by maximizing ELBO
        self.inference = ed.KLqp({self.U: self.qU, self.V: self.qV, self.U_prime: self.qU_prime, self.V_prime: self.qV_prime}, data={self.R_hats: self.ratings_selected})
        
    def sampling_graph(self):
        # Retrieve for specific user
        self.test_user_index = tf.placeholder(tf.int32)
        self.num_samples = tf.placeholder(tf.int32)
        
        # Collect samples for each latent factor related to user
        self.U_samples = self.U.sample(self.num_samples)
        self.U_samples_selected = tf.gather(tf.transpose(self.U_samples, [1, 0, 2]), self.test_user_index)
        self.U_samples_full = tf.reshape(tf.tile(self.U_samples_selected, [1, self.num_items]), [-1, self.D])
        
        self.U_prime_samples = self.U_prime.sample(self.num_samples)
        self.U_prime_samples_selected = tf.gather(tf.transpose(self.U_prime_samples, [1, 0, 2]), self.test_user_index)
        self.U_prime_samples_full = tf.reshape(tf.tile(self.U_samples_selected, [1, self.num_items]), [-1, self.D])
        
        self.V_samples = self.V.sample(self.num_samples)
        self.V_samples_full = tf.reshape(self.V_samples, [-1, self.D])
    
        self.V_prime_samples = self.V_prime.sample(self.num_samples)
        self.V_prime_samples_full = tf.reshape(self.V_prime_samples, [-1, self.D])
        
        # Pass samples through NNMF
        self.sample_decoder = NNMFCore(self.U_samples_full, self.V_samples_full, self.U_prime_samples_full, self.V_prime_samples_full, self.D, self.D_prime, sampling=True)
        self.R_mean_samples = tf.reshape(self.sample_decoder.prediction, [-1,self.num_items])
        self.R_samples = ed.models.Normal(loc=self.R_mean_samples, scale=0.3*tf.ones(tf.shape(self.R_mean_samples)))
        
    def build_graph(self):
        self.variational_layer()
        self.latent_factors()
        self.feed_forward()
        self.posterior_inference()
        self.sampling_graph()         
    
    def train(self, t_mask, sess=ed.get_session(), n_iter=2000):
        info_dicts = []
        user_indices, item_indices = np.array(np.where(t_mask))
        
        feed_dict = {
            self.user_indices: user_indices,
            self.item_indices: item_indices,
            self.ratings: self.dataset
            }
            
        for _ in range(n_iter):
            info_dict = self.inference.update(feed_dict=feed_dict)
            info_dicts.append(info_dict)
        losses = [x['loss'] for x in info_dicts]
        
        plt.plot(losses)
        plt.title('loss curve')
        plt.show()
        
        return losses

    def sample_user_ratings(self, user_index, num_samples=100, sess=ed.get_session()):
        feed_dict = {self.test_user_index: user_index, self.num_samples: num_samples}
        return sess.run(self.R_samples, feed_dict=feed_dict)
    


        
    
    
    
            