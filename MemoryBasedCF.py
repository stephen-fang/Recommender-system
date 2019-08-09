from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances

import pandas as pd
import numpy as np
# import cufflinks as cf
# import q
# cf.set_config_file(offline=True)


class MemoryBCF:
    def __init__(self, rating_path, type='user'):
        self.type = type
        self.rating_path = rating_path
        self.rating_train_M, self.rating_test_M = self._get_Matrix()
        self.result = self.evaluate()

    def _get_Matrix(self):
        rating = pd.read_csv(self.rating_path)
        rating_train = rating.sample(frac=0.7)
        rating_test = rating.drop(rating_train.index)
        rating_train_M = rating_train.pivot_table(index=['movieId'], columns=[
                                                  'userId'], values='rating').T.reset_index(drop=True)
        rating_train_M.fillna(0, inplace=True)
        rating_test_M = rating_test.pivot_table(index=['movieId'], columns=[
                                                'userId'], values='rating').T.reset_index(drop=True)
        rating_test_M.fillna(0, inplace=True)
        return rating_train_M, rating_test_M

    def _sim(self, M):
        if self.type == 'user':
            sim = pairwise_distances(M, metric='cosine')
        elif self.type == 'item':
            sim = pairwise_distances(M.T, metric='cosine')
        return sim

    def predict(self, rating, similarity):
        if self.type == 'user':
            mean_user_rating = rating.mean(axis=1)
            rating_diff = (rating - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(
                rating_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        elif self.type == 'item':
            pred = rating.dot(similarity) / \
                np.array([np.abs(similarity).sum(axis=1)])
        return np.array(pred)

    def rmse(self, pred, actual):
        actual = actual.values
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return np.sqrt(mean_squared_error(pred, actual))

    def evaluate(self):
        sim = self._sim(self.rating_train_M)
        pred = self.predict(self.rating_train_M, sim)
        rmse_val = self.rmse(pred, self.rating_test_M)
        return rmse_val


rating_path = 'dataset/rating.csv'
    
UserBM = MemoryBCF(rating_path, 'user')
ItemBM = MemoryBCF(rating_path, 'item')
print(f'User based RMSE: {UserBM.result}')
print(f'Item based RMSE: {ItemBM.result}')