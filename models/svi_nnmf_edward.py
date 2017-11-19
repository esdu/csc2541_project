import numpy as np
import tensorflow as tf
import edward as ed
from matplotlib import pyplot as plt

# turn off trainable for nn parameters
class NNMFCore:
    def __init__(self, U, V, U_prime, V_prime, user_indices, item_indices, D=2, D_prime=2, hidden_units_per_layer=4):
        self.U, self.V, self.U_prime, self.V_prime = U, V, U_prime, V_prime
        self.D = D
        self.D_prime = D_prime
        self.nhid = hidden_units_per_layer 
        self.user_indices = user_indices    
        self.item_indices = item_indices  
        self.nn_parameters = []
            
    def nn_inputs_preprocess(self):
        U_selected = tf.gather_nd(self.U, self.user_indices)
        V_selected = tf.gather_nd(self.V, self.item_indices)
        U_prime_selected = tf.gather_nd(self.U_prime, self.user_indices)
        V_prime_selected = tf.gather_nd(self.V_prime, self.item_indices)        
        self.nn_inputs = tf.concat([U_selected, V_selected, tf.multiply(U_prime_selected, V_prime_selected)], 1)
        
    def nn_fclayers(self):
        
        with tf.name_scope('fc_layers') as scope:
            #fc1
            self.fc1W = tf.Variable(tf.random_normal([2*self.D + self.D_prime, self.nhid], dtype=tf.float32, stddev=1e-1), name='fc1_weights')
            self.fc1b = tf.Variable(tf.random_normal([self.nhid], dtype=tf.float32, stddev=1e-1), name='fc1_bias')
            fc1_in = tf.nn.bias_add(tf.matmul(self.nn_inputs, self.fc1W), self.fc1b)
            fc1 = tf.nn.sigmoid(fc1_in)
            self.nn_parameters += [self.fc1W, self.fc1b]
            
            #fc2
            self.fc2W = tf.Variable(tf.random_normal([self.nhid, self.nhid], dtype=tf.float32, stddev=1e-1), name='fc2_weights')
            self.fc2b = tf.Variable(tf.random_normal([self.nhid], dtype=tf.float32, stddev=1e-1), name='fc2_bias')
            fc2_in = tf.nn.bias_add(tf.matmul(fc1, self.fc2W), self.fc2b)
            fc2 = tf.nn.sigmoid(fc2_in)
            self.nn_parameters += [self.fc2W, self.fc2b]
            
            #fc3
            self.fc3W = tf.Variable(tf.random_normal([self.nhid, 1], dtype=tf.float32, stddev=1e-1), name='fc3_weights')
            self.fc3b = tf.Variable(tf.random_normal([1], dtype=tf.float32, stddev=1e-1), name='fc3_bias')
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
    def __init__(self, num_users, num_items, D=2, D_prime=2, mapping=NNMFCore):
        self.num_users, self.num_items = num_users, num_items
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
        self.user_indices = tf.placeholder(tf.int32, shape = [None, 1])
        self.item_indices = tf.placeholder(tf.int32, shape = [None, 1])
        self.ratings = tf.placeholder(tf.float32, shape = [None, None])  
        self.decoder = decoder(self.U, self.V, self.U_prime, self.V_prime, self.user_indices, self.item_indices)
        self.decoder.build_graph()
        
    def posterior_inference(self):
        self.R_hats = ed.models.Normal(loc=self.decoder.prediction, scale=0.3*tf.ones(tf.shape(self.decoder.prediction)))
        ratings_indices = tf.concat([self.user_indices, self.item_indices],1)
        self.ratings_selected = tf.gather_nd(self.ratings, ratings_indices)
        
        # posterior inference by
        
        self.R_hats_samples = self.R_hats.value() 
        self.inference = ed.KLqp({self.U: self.qU, self.V: self.qV, self.U_prime: self.qU_prime, self.V_prime: self.qV_prime}, data={self.R_hats: self.ratings_selected})
        
    def build_graph(self):
        self.variational_layer()
        self.latent_factors()
        self.feed_forward()
        self.posterior_inference()
        #self.posterior_samples()
    
    # def posterior_samples(self):
    #     self.U_samples = self.U.sample(100) 
    #     self.V_samples = self.V.sample(100)
    #     self.U_prime_samples = self.U_prime.sample(100)
    #     self.V_prime_samples = self.V_prime.sample(100)
    #     self.sample_decoder = 
    #     R_mean_samples=
         
    
def train(graph, user_indices, item_indices, ratings, sess, n_iter=300000):
    info_dicts = []
    feed_dict = {
        graph.user_indices: user_indices,
        graph.item_indices: item_indices,
        graph.ratings: ratings
        }
    for _ in range(n_iter):
        info_dict = graph.inference.update(feed_dict=feed_dict)
        info_dicts.append(info_dict)
    losses = [x['loss'] for x in info_dicts]
    return losses

#def predict(graph, user_indices, item_indices):
    


        
    
    
    
            