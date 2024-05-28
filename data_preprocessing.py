from global_imports import *

#importing the training data
imdb_data=pd.read_csv('IMDB Dataset.csv')
print("Data shape...: ",imdb_data.shape)
print(f"glimpse of data: {imdb_data.head(10)}")