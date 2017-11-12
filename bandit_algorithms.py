from util import make_bandits_mask

import numpy as np
import random

class BanditAlgorithms:
    def __init__(self, user, R, mask, gamma, model, retrain):

        self.user = user
        self.R = R
        self.mask = mask
        self.gamma = gamma

        self.model = model
        self.retrain = retrain

    def get_egreedy_score(self, epsilon):
        # TODO: retraining is prohibitively expensive here...
        
        my_mask = make_bandits_mask(self.mask, self.R) # Note: This returns a copy of mask.
        discount = 1
        score = 0
        
        n_iter = np.sum(my_mask[self.user] == 0)
        
        for _ in range(n_iter):
            user_ratings = self.model.posterior_map[self.user]
            user_indices = np.array(range(len(user_ratings)))
            user_ratings = user_ratings[my_mask[self.user] == 0]
            user_indices = user_indices[my_mask[self.user] == 0]

            if random.random() < epsilon:
                # Choose greedy action
                idx = np.argmax(user_ratings)
            else:
                # Choose random action
                idx = random.randint(0,len(user_ratings)-1)

            my_mask[self.user,user_indices[idx]] = 1
            score += self.R[self.user,user_indices[idx]] * discount
            discount *= self.gamma

            if self.retrain:
                self.model.train(R=self.R, mask=my_mask, n_iter=5000)
        
        return score



    def get_ucb_score(self):
        # TODO: retraining is prohibitively expensive here...
        
        my_mask = make_bandits_mask(self.mask, self.R) # Note: This returns a copy of mask.
        discount = 1
        score = 0
        
        n_iter = np.sum(my_mask[self.user] == 0)
        
        for i in range(n_iter):
            user_ratings = self.model.posterior_map[self.user]
            user_dev = np.std(self.model.posterior[self.user], axis=0)
            user_indices = np.array(range(len(user_ratings)))
            user_ratings = user_ratings[my_mask[self.user] == 0]
            user_dev = user_dev[my_mask[self.user] == 0]
            user_indices = user_indices[my_mask[self.user] == 0]
            
            idx = np.argmax(user_ratings + user_dev)
            
            my_mask[self.user,user_indices[idx]] = 1
            score += self.R[self.user,user_indices[idx]] * discount
            discount *= self.gamma   

            if self.retrain:
                self.model.train(R=self.R, mask=my_mask, n_iter=5000)
                
        return score

    def get_best_score(self):
        '''
        user: user index
        R: true ratings matrix of size (n_users, n_items)
        mask: mask matrix of size (n_users, n_items), 1 if ratings is observed, 0 otherwise.
        '''
        mask_bandit = make_bandits_mask(self.mask, self.R)
        # Extract missing entries
        missing_ratings = self.R[self.user][np.where(mask_bandit[self.user] == 0)]
        # Sort biggest to smallest
        missing_ratings = missing_ratings[np.argsort(-missing_ratings)]
        best_score = 0
        discount = 1
        for r in missing_ratings:
            best_score += r * discount
            discount *= self.gamma
        return best_score