import pandas as pd


class Load:
    def __init__(self, dataset='small'):
        path = '/Users/xuanxinfang/Dropbox/UNSW/T2/COMP9417/assignment/dataset'
        ratings = pd.read_csv(path+"/rating.csv")
        movies = pd.read_csv(path+"/movie.csv")
        
