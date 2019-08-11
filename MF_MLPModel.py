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
import matplotlib.pyplot as plt

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
        n_latent_factors_user = 10
        n_latent_factors_movie = 10
        n_latent_factors_mf = 5
        dropout_rate = 0.3
        # n_users, n_movies = len(rating.userId.unique()), len(
        #     rating.movieId.unique())

        movie_input = keras.layers.Input(shape=[1], name='Item')
        movie_embedding_mlp = keras.layers.Embedding(
            num_movies + 1, n_latent_factors_movie, name='Movie-Embedding-MLP')(movie_input)
        movie_vec_mlp = keras.layers.Flatten(
            name='FlattenMovies-MLP')(movie_embedding_mlp)
        movie_vec_mlp = keras.layers.Dropout(dropout_rate)(movie_vec_mlp)

        movie_embedding_mf = keras.layers.Embedding(
            num_movies + 1, n_latent_factors_mf, name='Movie-Embedding-MF')(movie_input)
        movie_vec_mf = keras.layers.Flatten(
            name='FlattenMovies-MF')(movie_embedding_mf)
        movie_vec_mf = keras.layers.Dropout(dropout_rate)(movie_vec_mf)

        user_input = keras.layers.Input(shape=[1], name='User')
        user_vec_mlp = keras.layers.Flatten(name='FlattenUsers-MLP')(keras.layers.Embedding(
            num_users + 1, n_latent_factors_user, name='User-Embedding-MLP')(user_input))
        user_vec_mlp = keras.layers.Dropout(dropout_rate)(user_vec_mlp)

        user_vec_mf = keras.layers.Flatten(name='FlattenUsers-MF')(keras.layers.Embedding(
            num_users + 1, n_latent_factors_mf, name='User-Embedding-MF')(user_input))
        user_vec_mf = keras.layers.Dropout(dropout_rate)(user_vec_mf)

        #concat = keras.layers.merge([movie_vec_mlp, user_vec_mlp], mode='concat',name='Concat')
        concat = concatenate(
            [movie_vec_mlp, user_vec_mlp], axis=-1, name='Concat')

        concat_dropout = keras.layers.Dropout(dropout_rate)(concat)
        dense = keras.layers.Dense(
            200, name='FullyConnected')(concat_dropout)
        dense_batch = keras.layers.BatchNormalization(name='Batch')(dense)
        dropout_1 = keras.layers.Dropout(
            dropout_rate, name='Dropout-1')(dense_batch)
        dense_2 = keras.layers.Dense(
            100, name='FullyConnected-1')(dropout_1)
        dense_batch_2 = keras.layers.BatchNormalization(
            name='Batch-2')(dense_2)

        dropout_2 = keras.layers.Dropout(
            dropout_rate, name='Dropout-2')(dense_batch_2)
        dense_3 = keras.layers.Dense(
            50, name='FullyConnected-2')(dropout_2)
        dense_4 = keras.layers.Dense(
            20, name='FullyConnected-3', activation='relu')(dense_3)

        #pred_mf = keras.layers.merge([movie_vec_mf, user_vec_mf], mode='dot',name='Dot')
        pred_mf = dot([movie_vec_mf, user_vec_mf], axes=-1, name='Dot')

        pred_mlp = keras.layers.Dense(
            1, activation='relu', name='Activation')(dense_4)

        #combine_mlp_mf = keras.layers.merge([pred_mf, pred_mlp], mode='concat',name='Concat-MF-MLP')
        combine_mlp_mf = concatenate(
            [pred_mf, pred_mlp], axis=-1, name='Concat-MF-MLP')

        result_combine = keras.layers.Dense(
            100, name='Combine-MF-MLP')(combine_mlp_mf)
        deep_combine = keras.layers.Dense(
            100, name='FullyConnected-4')(result_combine)

        result = keras.layers.Dense(1, name='Prediction')(deep_combine)

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
        y_test_hat = self.model.predict([self.test.userId, self.test.movieId])
        err = np.sqrt(mean_squared_error(y_test_true, y_test_hat))
        print(f'MF-MLP model test err = {err}')

    def pplot(self):
        pd.Series(self.history.history['loss']).plot(logy=True)
        plt.xlabel("Epoch")
        plt.ylabel("Train Error")

    # def why(self):
    #     y_true = self.test.rating
    #     y_hat_2 = np.round(self.model.predict(
    #         [self.test.userId, self.test.movieId]), 0)
    #     print(mean_squared_error(y_true, y_hat_2))

    #     print(mean_squared_error(y_true, self.model.predict(
    #         [self.test.userId, self.test.movieId])))
M = MFMLPModel()
M.eval()
# M.why()
