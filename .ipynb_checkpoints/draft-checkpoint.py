import pandas as pd

rating_path = '/Users/xuanxinfang/Dropbox/UNSW/T2/COMP9417/assignment/dataset/rating.csv'
movie_path = '/Users/xuanxinfang/Dropbox/UNSW/T2/COMP9417/assignment/dataset/movie.csv'
tag_path = '/Users/xuanxinfang/Dropbox/UNSW/T2/COMP9417/assignment/dataset/tag.csv'
link_path = '/Users/xuanxinfang/Dropbox/UNSW/T2/COMP9417/assignment/dataset/link.csv'

ratings = pd.read_csv(rating_path)
# print(rating.head(10))
movies = pd.read_csv(movie_path)
# print(movies.head(10))
