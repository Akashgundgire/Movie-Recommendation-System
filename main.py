import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie dataset
st.subheader('welcome to movie recommendation system')
movie = pd.read_csv('main_data (1).csv')

# Create a CountVectorizer and compute the cosine similarity matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(movie['comb'])
similarity = cosine_similarity(count_matrix)

# Define the recommend function
def recommend(movie_name):
    try:
        index = movie[movie['movie_title'] == movie_name].index[0]
        lst = list(enumerate(similarity[index]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]  # excluding the first item since it is the requested movie itself
        recommended_movies = []

        for i in range(len(lst)):
            a = lst[i][0]
            recommended_movies.append(movie['movie_title'][a])

        dff = movie[movie['movie_title'].isin(recommended_movies)]

        for index, row in dff.iterrows():
            st.text(f'Director Name: {row["director_name"]}')
            st.text(f'Genres: {row["genres"]}')
            st.text(f'Movie Title: {row["movie_title"]}')
            st.text("----")

    except IndexError:
        st.warning("That movie is not present, please try again.")

# Get user input for movie name
movie_name = st.selectbox('Select a Movie', movie['movie_title'])


# Display the button
if st.button('Click to Get Recommendations'):
    # Call the recommend function only if a movie name is provided and the button is clicked
    if movie_name:
        recommend(movie_name)
def get_movies_by_director(director_name):
    director_name = director_name.strip()  # Remove leading/trailing whitespaces
    movies = movie[movie['director_name'].str.contains(director_name, case=False)]
    return movies['movie_title'].tolist()

# Get user input for director name
director_name_input= st.selectbox('Select a director_name', movie['director_name'])
director_name_input = director_name_input.title()  # Convert to title case for case-insensitive matching

# Display the button
if st.button('Click to Get Movies'):
    # Call the function only if a director name is provided and the button is clicked
    if director_name_input:
        movies_by_director = get_movies_by_director(director_name_input)

        if movies_by_director:
            st.write(f"Movies directed by {director_name_input}:")
            for movie_title in movies_by_director:
                st.write(movie_title)
        else:
            st.warning(f"No movies found for director {director_name_input}.")
    else:
        st.warning("Please enter a director name.")
