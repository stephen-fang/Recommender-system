# %%
import cufflinks as cf
import pandas as pd
cf.go_offline()


# %%
rating_path = 'dataset/rating.csv'
rating = pd.read_csv(rating_path)
rating.head(3)

# %%
rating['rating'].iplot(kind='histogram', histnorm='percent')

# %%
mean_rating_per_user = rating.groupby(['userId']).mean()['rating']
mean_rating_per_user = mean_rating_per_user.apply(lambda x: round(x, 1))
mean_rating_per_user.iplot(kind='histogram', histnorm='percent')


#%%
mean_rating_per_movie = rating.groupby(['movieId']).mean()['rating']
mean_rating_per_movie = mean_rating_per_movie.apply(lambda x: round(x,0))
print(mean_rating_per_movie)
mean_rating_per_movie.iplot(kind='histogram', histnorm='percent')


#%%
