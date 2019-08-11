import pandas as pd
import keras
import numpy as np
import pandas as pd
from keras.layers import dot, concatenate
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras import optimizers
import logging
import os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class MFMLPModel:
    def __init__(self, rating_path="dataset/rating.csv"):
        self.rating_path = rating_path
        self.rating = self.read_data()

    def read_data(self):
        rating = pd.read_csv(self.rating_path)
        # start from 0
        rating.userId = rating.userId.astype('category').cat.codes.values
        rating.movieId = rating.movieId.astype('category').cat.codes.values
        return rating

    def build_model(self, latent=8, learning_rate=0.01):
        latent_MF = 3
        num_users, num_movies = len(
            self.rating.userId.unique()), len(self.rating.movieId.unique())
        # MF part
        # embedding user and flatten
        user_input_MF = keras.layers.Input(shape=[1, ], name='user_MF')
        # add bias
        user_embedding_MF = keras.layers.Embedding(
            num_users + 1, latent_MF, name='user_embedding_MF')(user_input_MF)
        # flatten user embedding

        user_vec_MF = keras.layers.Flatten()(user_embedding_MF)
        user_vec_MF = keras.layers.Dropout(0.2)(user_vec_MF)
        # movie part
        movie_input_MF = keras.layers.Input(shape=[1, ], name='movie_MF')
        movie_embedding_MF = keras.layers.Embedding(
            num_movies + 1, latent_MF, name='movie_embedding_MF')(movie_input_MF)
        movie_vec_MF = keras.layers.Flatten()(movie_embedding_MF)
        movie_vec_MF = keras.layers.Dropout(0.2)(movie_vec_MF)
        pred_MF = dot([movie_vec_MF, user_vec_MF], axes=-1)

        # MLP part
        user_input = keras.layers.Input(shape=[1, ], name='User')
        user_embedding = keras.layers.Embedding(
            num_users + 1, latent, name='user_embedding')(user_input)
        user_vec = keras.layers.Flatten()(user_embedding)
        user_vec = keras.layers.Dropout(0.2)(user_vec)

        movie_input = keras.layers.Input(shape=[1, ], name='Item')
        movie_embedding = keras.layers.Embedding(
            num_movies + 1, latent, name='movie_embedding')(movie_input)
        movie_vec = keras.layers.Flatten()(movie_embedding)
        movie_vec = keras.layers.Dropout(0.2)(movie_vec)

        concat = concatenate([movie_vec, user_vec], axis=-1, name='concat')
        concat = keras.layers.Dropout(0.2)(concat)

        output = keras.layers.Dense(200, name='FullyConnected_1')(concat)
        output = keras.layers.BatchNormalization(name='Batch_1')(output)
        output = keras.layers.Dropout(0.2, name='Dropout_1')(output)

        output = keras.layers.Dense(100, name='FullyConnected_2')(output)
        output = keras.layers.BatchNormalization(name='Batch_2')(output)
        output = keras.layers.Dropout(0.2, name='Dropout_2')(output)

        output = keras.layers.Dense(50, name='FullyConnected_3')(output)
        output = keras.layers.Dense(
            20, name='FullyConnected_4', activation='relu')(output)

        #pred_mf = keras.layers.merge([movie_vec_mf, user_vec_mf], mode='dot',name='Dot')

        pred = keras.layers.Dense(
            1, activation='relu', name='Activation')(output)

        combination = concatenate([pred_MF, pred], axis=-1, name='combination')

        result = keras.layers.Dense(100, name='FullyConnected_5')(pred_MF)
        result = keras.layers.Dense(100, name='FullyConnected_6')(result)

        result = keras.layers.Dense(1, name='Prediction')(result)

        self.model = keras.Model([user_input, movie_input], result)
        opt = keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, ts=0.2):
        self.build_model()
        self.train, self.test = train_test_split(self.rating, test_size=ts)
        self.history = self.model.fit(
            [self.train.userId, self.train.movieId], self.train.rating, epochs=25, verbose=0, validation_split=0.1)

    def eval(self):
        self.train()
        # y_hat_2 = np.round(model.predict([test.userId, test.movieId]),0)

        y_test_true = self.test.rating
        y_test_hat = np.round(self.model.predict(
            [self.test.userId, self.test.movieId]), 0)
        err = np.sqrt(mean_squared_error(y_test_true, y_test_hat))
        print(f'MF-MLP model test err = {err}')
        

# M = MFMLPModel()
# M.eval()
