import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests  # For fetching movie posters

# OMDb API Key (replace with your own key)
OMDB_API_KEY = '34c1bf3c'

# Function to fetch movie poster from OMDb API
def fetch_movie_poster(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if 'Poster' in data and data['Poster'] != 'N/A':
        return data['Poster']
    return None

# Load the preprocessed Netflix dataset
@st.cache_data
def load_data():
    df = pd.read_csv('netflix_titles.csv')  # Ensure the file path is correct
    return df

# Preprocess and combine features for content-based filtering
def preprocess_data(df):
    df['director'] = df['director'].fillna('')
    df['cast'] = df['cast'].fillna('')
    df['listed_in'] = df['listed_in'].fillna('')
    df['description'] = df['description'].fillna('')
    
    # Combine features
    df['combined_features'] = df['type'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['listed_in'] + ' ' + df['description']
    return df

# Build the recommendation system (content-based)
def build_recommender(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Get content-based recommendations
def get_content_recommendations(title, df, cosine_sim):
    try:
        idx = df[df['title'].str.contains(title, case=False)].index[0]
    except IndexError:
        return "Title not found!"
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Main application
def main():
    st.title('Netflix Movie Recommendation System')

    # Load and preprocess the data
    df = load_data()
    df = preprocess_data(df)
    cosine_sim = build_recommender(df)

    # Get user input
    movie_title = st.text_input('Enter a movie title:')

    if st.button('Get Recommendations'):
        if movie_title:
            recommendations = get_content_recommendations(movie_title, df, cosine_sim)
            if isinstance(recommendations, str):
                st.write(recommendations)
            else:
                st.write('Top 10 Recommendations:')
                cols = st.columns(3)  # Create 4 columns for the grid
                for i, movie in enumerate(recommendations):
                    with cols[i % 3]:  # Place the movie in the corresponding column
                        st.write(f"**{movie}**")
                        
                        # Fetch and display the poster
                        poster_url = fetch_movie_poster(movie)
                        if poster_url:
                            st.image(poster_url, width=200)  # Display the movie poster
                        else:
                            st.write("Poster not available.")
        else:
            st.write('Please enter a valid movie title.')

if __name__ == '__main__':
    main()
