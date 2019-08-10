import pandas as pd
import keras
import numpy as np
import pandas as pd
from keras.layers import dot
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras import optimizers
import warnings
warnings.filterwarnings('ignore')
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class MFModel:
    def __init__(self, rating_path="dataset/rating.csv"):
        self.rating_path = rating_path
        self.rating = self.read_data()

    def read_data(self):
        rating = pd.read_csv(self.rating_path)
        # start from 0
        rating.userId = rating.userId.astype('category').cat.codes.values
        rating.movieId = rating.movieId.astype('category').cat.codes.values
        return rating

    def build_model(self, latent=8, learning_rate=0.001):

        num_users, num_movies = len(
            self.rating.userId.unique()), len(self.rating.movieId.unique())

        # embedding user and flatten
        user_input = keras.layers.Input(shape=[1, ], name='user')
        # add bias
        user_embedding = keras.layers.Embedding(
            num_users + 1, latent, name='user_embedding')(user_input)
        # flatten user embedding

        user_vec = keras.layers.Flatten()(user_embedding)

        # movie part
        movie_input = keras.layers.Input(shape=[1, ], name='movie')
        movie_embedding = keras.layers.Embedding(
            num_movies + 1, latent, name='movie_embedding')(movie_input)
        movie_vec = keras.layers.Flatten()(movie_embedding)

        product = dot([movie_vec, user_vec], axes=-1)

        self.model = keras.Model([user_input, movie_input], product)

        opt = optimizers.SGD(lr=learning_rate, decay=1e-6,
                             momentum=0.9, nesterov=True)
        self.model.compile(optimizer=opt, loss='mean_squared_error')

    def train(self, ts=0.2):
        self.build_model()
        self.train, self.test = train_test_split(self.rating, test_size=ts)
        self.history = self.model.fit(
            [self.train.userId, self.train.movieId], self.train.rating, epochs=100, verbose=0)

    def eval(self):
        self.train()
        # y_hat_2 = np.round(model.predict([test.userId, test.movieId]),0)
        y_test_true = self.test.rating
        y_test_hat = np.round(self.model.predict(
            [self.test.userId, self.test.movieId]), 0)
        err = np.sqrt(mean_squared_error(y_test_true, y_test_hat))
        print(f'MF model test err = {err}')


M = MFModel()
M.eval()
