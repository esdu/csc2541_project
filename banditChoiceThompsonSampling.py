import numpy as np
import random

class BanditChoiceThompsonSampling(object):
    def __init__(self):
        pass

    def evaluate(self, posteriorMatrix, legalItemVector):
 
        user_indices = np.array(range(len(legalItemVector)))
        user_indices = user_indices[legalItemVector == 1]
        user_ratings = posteriorMatrix[:,user_indices]

        itemIndex = self.get_thompson_sample(user_ratings, user_indices)

        return itemIndex
    

    def get_thompson_sample(self, user_ratings, user_indices):
        #randomly sample ratings and take the argmax of samples

        random_sample = random.randint(0,len(user_ratings)-1)
        idx = np.argmax(user_ratings[random_sample])
        selected_item = user_indices[idx]
        return selected_item