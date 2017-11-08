"""
This class is for analyzing the movie lens dataset.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os # For reading files 

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
        self.userMovieRatingMatrix = self.parseUserMovieMatrix()

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

    def parseUserPreference(self):
        userPreferences = {} # Index each user using the ID as key
        with open(self.userFile) as userFile:
            for currLine in userFile:
                currLine = currLine.strip()
                if currLine:
                    userPreference = MovieLensUserPreference(*tuple(currLine.split('|')))
                    userPreferences[userPreference.id] = userPreference
        return userPreferences

    def parseUserMovieMatrix(self):
        with open(self.dataFile) as dataFile:
            # Get all the user and movie ratings into ids
            ratingMatrix = np.zeros((943, 1682)) # 943 users from 1 to 943, 1682 items based on dataset
            for currLine in dataFile:
                currLine = currLine.strip()
                if currLine:
                    singleRating = MovieLensRating(*tuple(currLine.split()))
                    ratingMatrix[singleRating.userId-1][singleRating.movieId-1] = singleRating.rating
            return ratingMatrix


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
    
    def __hash__(self):
        return hash((self.userId, self.movieId, self.rating, self.timeStamp))

if __name__ == "__main__": 
    movieLensAnalyzer = MovieLensAnalyzer()
    # Returns as a numpy array
    userMovieRatingMatrix = movieLensAnalyzer.getUserMovieRatingMatrix()
    print(userMovieRatingMatrix.shape)

