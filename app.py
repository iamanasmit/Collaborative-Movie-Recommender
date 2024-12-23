import pandas as pd

movies=pd.read_csv('movies.csv')
ratings=pd.read_csv('ratings.csv')

final_dataset=ratings.pivot(index='movieId', columns='userId', values='rating')
final_dataset.fillna(0, inplace=True)

no_users_voted=ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted=ratings.groupby('userId')['rating'].agg('count')

final_dataset=final_dataset.loc[no_users_voted[no_users_voted>10].index, :]
final_dataset=final_dataset.loc[:, no_movies_voted[no_movies_voted>50].index]

from scipy.sparse import csr_matrix
csr_data=csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


def get_recommendation(movie: str):
    recommedations = movies[movies['title'].str.contains(movie)]
    if len(recommedations) == 0:
        return 'not found'
    else:
        movie_idx = recommedations.iloc[0]['movieId']
        try:
            final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        except:
            return 'filtered out'
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        
        _, indices = model.kneighbors(csr_data[movie_idx], n_neighbors=11)
        
        indices = indices.flatten()
        recommedations = movies[movies['movieId'].isin(final_dataset.iloc[indices]['movieId'])]
        
        return recommedations

# recommendations=get_recommendation('Iron Man')
# print(recommendations['title'].tolist())

import streamlit as st

st.title('Movie Recommender System')
movie = st.text_input('Enter movie name')
if movie:
    recommendations = get_recommendation(movie)
    st.subheader('Recommendations')
    st.write(recommendations['title'].tolist())