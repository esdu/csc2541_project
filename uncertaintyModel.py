class UncertaintyModel(object):
    def __init__(self, ratingMatrix):
        self.ratingMatrix = ratingMatrix
        #self.posteriorMatrix = None

    def reset(self, seed=None):
        """
        Reset the weights as if no training was done.
        Reset seed.
        """
        pass
    
    def save(self, fname):
        return fname
    
    def load(self, fname):
        return True
    
    def train(self, legalTrainIndices):
        # Train the weights based on current legalTrainIndices matrix
        pass

    def sample_for_user(self, i):
        # return (k, m) matrix of k samples for user i
        return samples
