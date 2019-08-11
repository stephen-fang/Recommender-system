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
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.models import load_model


class MLPModel:
    def __init__(self, rating_path="dataset/rating.csv", epochs=25, latent=8, learning_rate=0.005):
        self.rating_path = rating_path
        self.rating = self.read_data()
        self.latent = latent
        self.learning_rate = learning_rate
        self.epochs = epochs
    def read_data(self):
        rating = pd.read_csv(self.rating_path)
        # start from 0
        rating.userId = rating.userId.astype('category').cat.codes.values
        rating.movieId = rating.movieId.astype('category').cat.codes.values
        return rating

    def build_model(self):
        latent = self.latent
        learning_rate = self.learning_rate
        num_users, num_movies = len(
            self.rating.userId.unique()), len(self.rating.movieId.unique())

        user_input = keras.layers.Input(shape=[1, ])
        user_embedding = keras.layers.Embedding(
            num_users + 1, latent)(user_input)
        user_vec = keras.layers.Flatten()(user_embedding)
        user_vec = keras.layers.Dropout(0.2)(user_vec)

        movie_input = keras.layers.Input(shape=[1, ])
        movie_embedding = keras.layers.Embedding(
            num_movies + 1, latent)(movie_input)
        movie_vec = keras.layers.Flatten()(movie_embedding)
        movie_vec = keras.layers.Dropout(0.2)(movie_vec)

        concat = concatenate([movie_vec, user_vec], axis=-1)

        concat = keras.layers.Dropout(0.2)(concat)
        output = keras.layers.Dense(256)(concat)
        output = keras.layers.BatchNormalization()(output)

        output = keras.layers.Dropout(0.2)(output)
        output = keras.layers.Dense(128)(output)
        output = keras.layers.BatchNormalization()(output)

        output = keras.layers.Dropout(0.3)(output)
        output = keras.layers.Dense(64)(output)
        output = keras.layers.Dense(16, activation='relu')(output)
        pred = keras.layers.Dense(
            1, activation='relu', name='Activation')(output)

        result = keras.layers.Dense(128)(pred)

        result = keras.layers.Dense(1, name='Prediction')(result)

        self.model = keras.Model([user_input, movie_input], result)
        opt = keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, ts=0.2):
        self.build_model()
        epch = self.epochs
        self.train, self.test = train_test_split(self.rating, test_size=ts)
        self.history = self.model.fit(
            [self.train.userId, self.train.movieId], self.train.rating, epochs=epch, verbose=0)

    def eval(self):
        # try:
        #     self.model = load_model('models/MLPModel.h5')
        # except:
        #     self.train()
        #     self.model.save('models/MLPModel.h5')
        self.train()
        # y_hat_2 = np.round(model.predict([test.userId, test.movieId]),0)
        self.train, self.test = train_test_split(self.rating, test_size=0.2)
        y_test_true = self.test.rating
        y_test_hat = self.model.predict([self.test.userId, self.test.movieId])
        err = np.sqrt(mean_squared_error(y_test_true, y_test_hat))
        print(f'MLP Model test err = {err}')

    def pplot(self):
        pd.Series(self.history.history['loss']).plot(logy=True)
        plt.xlabel("Epoch")
        plt.ylabel("Train Error")

    # def get_graph(self):
    #     self.svg = SVG(model_to_dot(model,  show_shapes=False,
    #                                 show_layer_names=True, rankdir='HB').create(prog='dot', format='svg'))
    #     return self.svg

# M = MLPModel()
# M.eval()
