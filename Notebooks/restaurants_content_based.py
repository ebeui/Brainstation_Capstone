import streamlit as st
import numpy as np
import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

# Ignore all warnings to avoid cluttering the output
import warnings
warnings.filterwarnings("ignore")

# Load data from a pickle file into a Pandas DataFrame
rec_data = pd.read_pickle('rec_data.pkl')

# Set up the Streamlit app
st.title('Restaurant Recommender')

# User input: Autocomplete dropdown prediction
restaurant_name_input = st.selectbox("Enter a restaurant name:", rec_data['restaurant_name'])

# Check if the restaurant name is in the dataset
if restaurant_name_input in rec_data['restaurant_name'].values:
    # Create a function to calculate recommendations
    def restaurant_recommender(name, restaurants, similarities):
        # Get the restaurant by name
        restaurant_data = restaurants[restaurants['restaurant_name'] == name]
        business_id = restaurant_data['business_id'].values[0]

        # Create a dataframe with the restaurant names and similarities
        sim_df = pd.DataFrame(
            {'restaurant': restaurants['restaurant_name'], 
             'similarity': similarities[business_id]
            })

        # Get the top 10 similar restaurants
        top_restaurants = sim_df.sort_values(by='similarity', ascending=False).head(10)

        return top_restaurants

    # Create the TF-IDF vectorizer
    ENGLISH_STOP_WORDS = stopwords.words('english')
    stemmer = PorterStemmer() 

    def tokenizer(sentence):
        # remove punctuation and set to lower case
        for punctuation_mark in string.punctuation:
            sentence = sentence.replace(punctuation_mark,'').lower()

        # split sentence into words
        listofwords = sentence.split(' ')
        listofstemmed_words = []

        # remove stopwords and any tokens that are just empty strings
        for word in listofwords:
            if (not word in ENGLISH_STOP_WORDS) and (word!=''):
                # Stem words
                stemmed_word = stemmer.stem(word)
                listofstemmed_words.append(stemmed_word)

        return listofstemmed_words

    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer, min_df=30, max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(rec_data['features'])

    # Calculate the cosine similarity matrix 
    cosine_similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Calculate recommendations
    recommendations = restaurant_recommender(restaurant_name_input, rec_data, cosine_similarity_matrix).reset_index(drop=True)
    recommendations.index += 1

    # Filter out the user's input restaurant from the recommendations
    recommendations = recommendations[recommendations['restaurant'] != restaurant_name_input]

    # Display recommendations with numbering
    st.write("Top 10 Recommended Restaurants:")
    st.dataframe(recommendations[['restaurant']].reset_index(drop=True))
else:
    st.write("Restaurant not found in the dataset.")
