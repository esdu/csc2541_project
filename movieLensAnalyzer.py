"""
This class is for analyzing the movie lens dataset.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os # For reading files
import random
import sys

def namestr(obj):
    namespace = globals()
    return [name for name in namespace if namespace[name] is obj]

def pprint(obj):
    # Assumes obj is a numpy array, matrix
    print(namestr(obj), obj.shape)
    print(obj)

class MovieLensAnalyzer(object):
    def __init__(self):
        # Note: Below only works for 100k dataset
        self.currentWorkingDirectory = os.path.abspath(os.path.dirname(__file__))
        self.dataDirectory = os.path.join(self.currentWorkingDirectory, 'ml-100k')
        self.dataFile = os.path.join(self.dataDirectory, 'u.data')
        self.genreFile = os.path.join(self.dataDirectory, 'u.genre')
        self.itemFile = os.path.join(self.dataDirectory, 'u.item')
        self.occupationFile = os.path.join(self.dataDirectory, 'u.occupation')
        self.userFile = os.path.join(self.dataDirectory, 'u.user')
        self.userPreferences= self.parseUserPreference()
        self.userMovieRatingMatrix, self.trainRatingMatrix, self.testRatingMatrix = self.parseUserMovieMatrix()

        # Generate matrices for bandits
        self.legalUserMovieRatingMatrix = self.generateLegalMoveMatrix(self.userMovieRatingMatrix)
        self.legalTrainRatingMatrix = self.generateLegalMoveMatrix(self.trainRatingMatrix)
        self.legalTestRatingMatrix = self.generateLegalMoveMatrix(self.testRatingMatrix)

        # Initialize default mask and labels
        self.mask = self.generateMask()
        self.labels = self.generatePositiveNegatives()

    def printMe(self):
        pprint(self.userMovieRatingMatrix)
        pprint(self.trainRatingMatrix)
        pprint(self.testRatingMatrix)
        pprint(self.legalUserMovieRatingMatrix)
        pprint(self.legalTrainRatingMatrix)
        pprint(self.legalTestRatingMatrix)
        pprint(self.mask)
        pprint(self.labels)

    def simplifyMatrix(self, numElements):
        """
        Make all matrices have numElements only for debugging
        """
        # TODO: Randomize below
        startIndex = 20
        endIndex = startIndex + int(numElements)
        self.userMovieRatingMatrix = self.userMovieRatingMatrix[startIndex:endIndex, startIndex:endIndex]
        # TODO: train and test do not correspond to actual userMovieRating Split
        self.trainRatingMatrix = self.trainRatingMatrix[startIndex:endIndex, startIndex:endIndex]
        self.testRatingMatrix = self.testRatingMatrix[startIndex:endIndex, startIndex:endIndex]
        self.legalUserMovieRatingMatrix= self.legalUserMovieRatingMatrix[startIndex:endIndex, startIndex:endIndex]
        self.legalTrainRatingMatrix = self.legalTrainRatingMatrix[startIndex:endIndex, startIndex:endIndex]
        self.legalTestRatingMatrix = self.legalTestRatingMatrix[startIndex:endIndex, startIndex:endIndex]
        self.mask = self.mask[startIndex:endIndex, startIndex:endIndex]
        self.labels= self.labels[startIndex:endIndex, startIndex:endIndex]

    def generateMask(self, percentageZero=0.3):
        '''
        Generates the mask for the userMovieRatingMatrix
        '''
        mask = np.ones(self.userMovieRatingMatrix.shape)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (random.random() < float(percentageZero)):
                    mask[i][j] = 0
        return mask

    def getMaskedUserMovieRatingMatrix(self):
        if self.mask is None:
            raise Exception("Masked not initialize using generateMask()")
        return self.userMovieRatingMatrix * self.mask

    def getLegalMatrix(self):
        return self.legalUserMovieRatingMatrix, self.legalTrainRatingMatrix, self.legalTestRatingMatrix

    def getUserMovieRatingMatrix(self):
        return self.userMovieRatingMatrix, self.trainRatingMatrix, self.testRatingMatrix

    # Evaluation: rmse
    def rootMeanSquareError(self, reconstructionMatrix):
        return np.sqrt(np.mean((reconstructionMatrix - self.userMovieRatingMatrix)**2))

    # Evaluation: meanAveragePrecision
    def meanAveragePrecision(self, reconstructionMatrix):
        # TODO: Calculate MAP for ratings
        return 0

    # TODO: Evaluate regret as suggested by WeiZhen

    # Evaluate regret as suggested by Eddie
    def optimalExplorationCalculation(self, discountFactor, legalMoveMatrix, ratingMatrix):
        # Calculate best score for each user discounted by discount factor
        sortedMatrix = ratingMatrix.copy()
        sortedMatrix[np.where(legalMoveMatrix != 1)] = 0
        # Sort each user's rating
        sortedMatrix.sort(axis=1)
        # Sort from increasing to decreasing
        sortedMatrix = sortedMatrix[:,::-1]

        powers = np.array([range(ratingMatrix.shape[1]) for _ in range(ratingMatrix.shape[0])])
        regretMatrix = np.ones(ratingMatrix.shape)
        # Make discount factor the initial values
        regretMatrix *= discountFactor
        discounted = np.power(regretMatrix, powers)
        optimalMatrix = sortedMatrix * discounted
        optimalResult = np.sum(optimalMatrix, 1)
        '''
        pprint(sortedMatrix)
        pprint(discounted)
        pprint(optimalMatrix)
        pprint(optimalResult)
        '''
        return optimalResult

    def parseUserPreference(self):
        userPreferences = {} # Index each user using the ID as key
        with open(self.userFile) as userFile:
            for currLine in userFile:
                currLine = currLine.strip()
                if currLine:
                    userPreference = MovieLensUserPreference(*tuple(currLine.split('|')))
                    userPreferences[userPreference.id] = userPreference
        return userPreferences

    def generatePositiveNegatives(self, positiveThresholdRating=3):
        # Divide each cell in userRatingMatrix into
        # either positive or negative based on the thresholdRating
        labels = np.zeros(self.userMovieRatingMatrix.shape)
        labels[np.where(self.userMovieRatingMatrix >= positiveThresholdRating)] = 1
        return labels

    def parseUserMovieMatrix(self, trainTestSplit = 0.5):
        # Sort all the ratings by timestamps
        arr = list()
        with open(self.dataFile) as dataFile:
            for currLine in dataFile:
                currLine = currLine.strip()
                if currLine:
                    singleRating = MovieLensRating(*tuple(currLine.split()))
                    arr.append(singleRating)
        # Sorted from earlier timestamp to later timestamp
        arr.sort()
        arr = np.array(arr)
        numRating = arr.size
        splitIndex = int(trainTestSplit*numRating)
        trainArr = arr[:splitIndex]
        testArr = arr[splitIndex:]
        trainRatingMatrix = np.zeros((943, 1682)) # 943 users from 1 to 943, 1682 items based on dataset
        testRatingMatrix = np.zeros((943, 1682)) # 943 users from 1 to 943, 1682 items based on dataset
        # Create a train matrix from earlier timestamps
        for trainRating in trainArr:
            trainRatingMatrix[trainRating.userId-1][trainRating.movieId-1] = trainRating.rating

        # Create test matrix from later timestamps
        for testRating in testArr:
            testRatingMatrix[testRating.userId-1][testRating.movieId-1] = testRating.rating

        # TODO: Handle case where a user only exist in train or test.
        #   FixIdea1: Just get intersection of users in train and test and delete those who arent in any of 2.

        # Combine both test and train using or operation to get final userRatingMatrix
        ratingMatrix = np.zeros((943, 1682)) # 943 users from 1 to 943, 1682 items based on dataset
        ratingMatrix = testRatingMatrix.copy() # Deep copy
        # Set wherever the later timestamps are 0 to earlier time stamps if those are 0
        ratingMatrix[np.where(testRatingMatrix == 0)] = trainRatingMatrix[np.where(trainRatingMatrix == 0)]
        return ratingMatrix, trainRatingMatrix, testRatingMatrix

    def generateLegalMoveMatrix(self, matrixForMask):
        """
        Positions where returned matrix are 1 are legal places to explore
        Note: Must be called before any mask is applied
        """
        legalMoveMatrix = np.zeros(matrixForMask.shape)
        # Places where the rating matrix was not 0 to begin with are legal
        # i.e. places where it's rated anywhere from [1, 5]
        legalMoveMatrix[np.where(matrixForMask != 0)] = 1
        return legalMoveMatrix

    def legalExploreIndices(self, matrixForMask):
        
        # Test matrix tells you where you are allowed to xplore
        print("TODO")

class MovieLensUserPreference(object):
    """
    Data for a movie lens user preference data
    """

    def __init__(self, id, age, gender, occupation, zipcode):
        self.id = int(id)
        self.age = int(age)
        self.gender = gender
        self.occupation = occupation
        self.zipcode = zipcode

    def __eq__(self, other):
        return isinstance(other, MovieLensUser) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

class MovieLensRating(object):
    """
    UserItemRating
    Represents a single row in user item matrix
    """

    def __init__(self, userId, movieId, rating, timeStamp):
        self.userId= int(userId)
        self.movieId = int(movieId)
        self.rating = int(rating)
        self.timeStamp = int(timeStamp)

    def __eq__(self, other):
        return (isinstance(other, MovieLensRating) and
                self.userId, self.movieId, self.rating, self.timeStamp ==
                other.userId, other.movieId, other.rating, other.timeStamp)

    # Use __lt__ for python3 compatibility.
    def __lt__(self, other):
        return self.timeStamp < other.timeStamp

    def __hash__(self):
        return hash((self.userId, self.movieId, self.rating, self.timeStamp))

if __name__ == "__main__":
    movieLensAnalyzer = MovieLensAnalyzer()
    movieLensAnalyzer.simplifyMatrix(5)
    # Get the user, train and test as numpy arrays
    userMovieRatingMatrix, trainRatingMatrix, testRatingMatrix = movieLensAnalyzer.getUserMovieRatingMatrix()
    legalUserMovieRatingMatrix, legalTrainRatingMatrix, legalTestRatingMatrix = movieLensAnalyzer.getLegalMatrix()
    #movieLensAnalyzer.printMe()
    '''
    percentageZero = 0.3
    movieLensAnalyzer.generateMask(percentageZero)
    # Anything rated 3 and above are labeled positive
    movieLensAnalyzer.generatePositiveNegatives(3)
    '''
    discountFactor = 0.99
    # Get optimal bandit calculation for each user
    optimalResult = movieLensAnalyzer.optimalExplorationCalculation(discountFactor, legalTrainRatingMatrix, trainRatingMatrix)

    '''
    if True:
        ratingMatrix = np.array([[1,2, 3], [4,5,6]])
        legalMoveMatrix = np.array([[0,1,1], [1,0,1]])
        pprint(ratingMatrix)
        pprint(legalMoveMatrix)
        sortedMatrix = ratingMatrix.copy()
        sortedMatrix[np.where(legalMoveMatrix != 1)] = 0
        # Sort each user's rating
        sortedMatrix.sort(axis=1)
        # Sort from increasing to decreasing
        sortedMatrix = sortedMatrix[:,::-1]

        powers = np.array([range(ratingMatrix.shape[1]) for _ in range(ratingMatrix.shape[0])])
        regretMatrix = np.ones(ratingMatrix.shape)
        # Make discount factor the initial values
        regretMatrix *= discountFactor
        discounted = np.power(regretMatrix, powers)
        optimalMatrix = sortedMatrix * discounted
        pprint(sortedMatrix)
        pprint(discounted)
        pprint(optimalMatrix)
    '''

