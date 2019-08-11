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
from keras.models import load_model

class MFMLPModel:
    def __init__(
        self,
        rating_path="dataset/rating.csv",
        latent=10,
        mf_latent=12,
        learning_rate=0.01,
        epochs=30,
    ):
        self.rating_path = rating_path
        self.rating = self.read_data()
        self.latent = latent
        self.mf_latent = mf_latent
        self.learning_rate = learning_rate
        self.epochs = epochs

    def read_data(self):
        rating = pd.read_csv(self.rating_path)
        # start from 0
        rating.userId = rating.userId.astype("category").cat.codes.values
        rating.movieId = rating.movieId.astype("category").cat.codes.values
        return rating

    def build_model(self):
        latent = self.latent
        mf_latent = self.mf_latent
        learning_rate = self.learning_rate
        num_users, num_movies = (
            len(self.rating.userId.unique()),
            len(self.rating.movieId.unique()),
        )
        dropout_rate = 0.3

        movie_input = keras.layers.Input(shape=[1])
        movie_embedding_mlp = keras.layers.Embedding(num_movies + 1, latent)(
            movie_input
        )
        movie_mlp = keras.layers.Flatten()(movie_embedding_mlp)
        movie_mlp = keras.layers.Dropout(dropout_rate)(movie_mlp)

        movie_embedding_mf = keras.layers.Embedding(num_movies + 1, mf_latent)(
            movie_input
        )
        movie_vec_mf = keras.layers.Flatten()(movie_embedding_mf)
        movie_vec_mf = keras.layers.Dropout(dropout_rate)(movie_vec_mf)

        user_input = keras.layers.Input(shape=[1])
        user_mlp = keras.layers.Flatten()(
            keras.layers.Embedding(num_users + 1, latent)(user_input)
        )
        user_mlp = keras.layers.Dropout(dropout_rate)(user_mlp)

        user_vec_mf = keras.layers.Flatten()(
            keras.layers.Embedding(num_users + 1, mf_latent)(user_input)
        )
        user_vec_mf = keras.layers.Dropout(dropout_rate)(user_vec_mf)

        concat = concatenate([movie_mlp, user_mlp], axis=-1)

        concat = keras.layers.Dropout(dropout_rate)(concat)
        output = keras.layers.Dense(200)(concat)
        output = keras.layers.BatchNormalization()(output)
        output = keras.layers.Dropout(dropout_rate)(output)
        output = keras.layers.Dense(100)(output)
        output = keras.layers.BatchNormalization()(output)

        output = keras.layers.Dropout(dropout_rate)(output)
        output = keras.layers.Dense(50)(output)
        output = keras.layers.Dense(20, activation="relu")(output)

        pred_mf = dot([movie_vec_mf, user_vec_mf], axes=-1)

        pred_mlp = keras.layers.Dense(1, activation="relu")(output)

        combine_mlp_mf = concatenate([pred_mf, pred_mlp], axis=-1)

        result = keras.layers.Dense(100)(combine_mlp_mf)
        result = keras.layers.Dense(100)(result)

        result = keras.layers.Dense(1)(result)

        self.model = keras.Model([user_input, movie_input], result)
        opt = keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(optimizer="sgd", loss="mean_squared_error")

    def train(self, ts=0.2):
        self.build_model()
        epch = self.epochs
        self.train, self.test = train_test_split(self.rating, test_size=ts)
        self.history = self.model.fit(
            [self.train.userId, self.train.movieId],
            self.train.rating,
            epochs=epch,
            verbose=0,
            validation_split=0.1,
        )

    def eval(self):
        # self.train()
        # try:
        #     self.model = load_model('models/MFMLPModel.h5')
        # except:
        #     self.train()
        #     self.model.save('models/MFMLPModel.h5')
        self.train()
        self.train, self.test = train_test_split(self.rating, test_size=0.2)
        # y_hat_2 = np.round(model.predict([test.userId, test.movieId]),0)

        y_test_true = self.test.rating
        y_test_hat = self.model.predict([self.test.userId, self.test.movieId])
        err = np.sqrt(mean_squared_error(y_test_true, y_test_hat))
        print(f"MF-MLP model test err = {err}")

    def pplot(self):
        pd.Series(self.history.history["loss"]).plot(logy=True)
        plt.xlabel("Epoch")
        plt.ylabel("Train Error")

    # def why(self):
    #     y_true = self.test.rating
    #     y_hat_2 = np.round(self.model.predict(
    #         [self.test.userId, self.test.movieId]), 0)
    #     print(mean_squared_error(y_true, y_hat_2))

    #     print(mean_squared_error(y_true, self.model.predict(
    #         [self.test.userId, self.test.movieId])))


# M = MFMLPModel()
# M.eval()
# M.why()
