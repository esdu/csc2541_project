"""
This class is for analyzing the movie lens dataset.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os # For reading files
import random
import sys

# import panda as pd , TODO: Later after fixing Ubuntu

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
        self.mask = None
        self.labels = None

    def simplifyMatrix(self, numElements):
        startIndex = 50
        endIndex = startIndex + int(numElements)
        self.userMovieRatingMatrix = self.userMovieRatingMatrix[startIndex:endIndex, startIndex:endIndex]

    def generateMask(self, percentageZero):
        '''
        Generates the mask for the userMovieRatingMatrix
        '''
        # TODO: Make this more efficient
        self.mask = np.ones(self.userMovieRatingMatrix.shape)
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                if (random.random() < float(percentageZero)):
                    self.mask[i][j] = 0

    def getMaskedUserMovieRatingMatrix(self):
        if self.mask is None:
            raise Exception("Masked not initialize using generateMask()")
        return self.userMovieRatingMatrix * self.mask

    def getUserMovieRatingMatrix(self):
        return self.userMovieRatingMatrix

        '''
        # TODO: Later after fixing Ubuntu, right now can't install anything including panda
        Code below found somewhere online
        ratings_list = [i.strip().split("::") for i in open(self.dataFile, 'r').readlines()]
        users_list = [i.strip().split("::") for i in open(self.userFile, 'r').readlines()]
        movies_list = [i.strip().split("::") for i in open(self.itemFile, 'r').readlines()]
        ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
        movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
        movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)
        '''

    # Evaluation: rmse
    def rootMeanSquareError(self, reconstructionMatrix):
        return np.sqrt(np.mean((reconstructionMatrix - self.userMovieRatingMatrix)**2))

    # Evaluation: meanAveragePrecision
    def meanAveragePrecision(self, reconstructionMatrix):
        # TODO: Calculate MAP
        return 0

    def parseUserPreference(self):
        userPreferences = {} # Index each user using the ID as key
        with open(self.userFile) as userFile:
            for currLine in userFile:
                currLine = currLine.strip()
                if currLine:
                    userPreference = MovieLensUserPreference(*tuple(currLine.split('|')))
                    userPreferences[userPreference.id] = userPreference
        return userPreferences

    def generatePositiveNegatives(self, positiveThresholdRating):
        # Divide each cell in userRatingMatrix into
        # either positive or negative based on the thresholdRating
        self.labels = np.zeros(self.userMovieRatingMatrix.shape)
        self.labels[np.where(self.userMovieRatingMatrix >= positiveThresholdRating)] = 1

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
        # Uncomment below after testing
        # ratingMatrix = testRatingMatrix.copy()
        # Check to make sure its deep copy.
        # ratingMatrix[np.where(testRatingMatrix == 0)] = trainRatingMatrix[np.where(trainRatingMatrix == 0)]
        ratingMatrix = np.logical_or(trainRatingMatrix, testRatingMatrix)

        return ratingMatrix, trainRatingMatrix, testRatingMatrix

    def legalExploreIndices(self):
        # TODO: Test matrix tells you where you are allowed to explore
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
    movieLensAnalyzer.simplifyMatrix(20)
    # Returns as a numpy array
    userMovieRatingMatrix = movieLensAnalyzer.getUserMovieRatingMatrix()
    movieLensAnalyzer.generateMask(0.3)
    haha = 0.3
    movieLensAnalyzer.generateMask(haha)
    # Anything rated 3 and above are labeled positive
    movieLensAnalyzer.generatePositiveNegatives(3)
    print(userMovieRatingMatrix.shape)

