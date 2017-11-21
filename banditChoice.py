import numpy as np

class BanditChoice(object):
    def __init__(self):
        pass

    def evaluate(self, posteriorMatrix, legalItemVector):
        # posteriorMatrix = (numSamples, numItems)
        # legalItemVector = (numItems)
        # itemIndex an integer within [0, numItems - 1]
        
        itemIndex = self.get_ucb(posteriorMatrix, legalItemVector)
        # itemIndex = self.get_egreedy(posteriorMatrix, legalItemVector)
        # itemIndex = self.get_thompson_sample(posteriorMatrix, legalItemVector)


        return itemIndex
    
    def get_ucb(self, user_ratings, user_indices):
        #get upper quantile of ratings    
        sorted_ratings = np.sort(user_ratings,axis=0)

        uquantile = int(len(user_ratings)*(3/4))-1
        uquantile_ratings = user_ratings[uquantile]

        idx = np.argmax(uquantile_ratings)
        selected_item = user_indices[idx]

        return selected_item


    def get_egreedy(self, user_ratings, user_indices, epsilon=0.1):
        # default to greedy if no e specified
        mean_ratings = np.mean(user_ratings,axis=0)
        if random.random() < 1-epsilon:
            idx = np.argmax(mean_ratings)
        else:
            idx = random.randint(0, len(mean_ratings)-1)
        
        return user_indices[idx]
        

    def get_thompson_sample(self, user_ratings, user_indices):
        random_sample = random.randint(0,len(user_ratings)-1)
        idx = np.argmax(user_ratings[random_sample])
        selected_item = user_indices[idx]
        return selected_item