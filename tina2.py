from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

def get_Matrix(rating_path):
    csv_df = pd.read_csv(rating_path)
    users_size = np.max(csv_df['userId'])
    movies_size = np.max(csv_df['movieId'])

    rating_M = np.zeros((users_size, movies_size))

    for row in csv_df.itertuples():
        rating_M[row[1]-1,row[2]-1] = row[3]

    return train_test_split(rating_M)
    
    
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test


train, test = get_Matrix('dataset/rating.csv')
print(train.shape, test.shape)

class MemoryBCF:
    def __init__(self, train, test, type='user'):
        self.type = type
        self.rating_train_M = train
        self.rating_test_M = test
        self.result = self.evaluate()

    def _sim(self, ratings, epsilon=1e-9):
        if self.type == 'user':
            sim = ratings.dot(ratings.T) + epsilon
        elif self.type == 'item':
            sim = ratings.T.dot(ratings) + epsilon
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return (sim / norms / norms.T)


    def _predict(self, ratings, similarity):
#         if self.type == 'user':
#             return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
#         elif self.type == 'item':
#             return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    
        
        pred = np.zeros(ratings.shape)
        k = 40
        if self.type == 'user':
            for i in range(ratings.shape[0]):
                top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
                for j in range(ratings.shape[1]):
                    pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                    pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
                #print(i)    
            return pred
        
        elif self.type == 'item':

            for j in range(ratings.shape[1]):
                top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
                for i in range(ratings.shape[0]):
                    pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                    pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))     
                print(j)
            return pred
        
    def _rmse(self, pred, actual):
        #actual = actual.values
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return np.sqrt(mean_squared_error(pred, actual))

    
    def evaluate(self):
        sim = self._sim(self.rating_train_M)
        pred = self._predict(self.rating_train_M, sim)
        rmse_val = self._rmse(pred, self.rating_test_M)
        return rmse_val
        
# UserBM = MemoryBCF(train, test, 'user') 
# print(f'User based RMSE: {UserBM.result}')

ItemBM = MemoryBCF(train, test, 'item')
print(f'Item based RMSE: {ItemBM.result}')
        