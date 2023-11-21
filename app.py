import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load data and models
df = pickle.load(open('restaurants.pkl', 'rb'))
recommendations = pickle.load(open('similarity.pkl', 'rb'))

unique_cuisines_df = pickle.load(open('Cuisine.pkl', 'rb'))
unique_cuisines_list = unique_cuisines_df['Cuisine'].tolist()

def recommend_restaurants(df, preferred_cuisines, budget, min_rating,choice,choice2):
    
    veg = choice
    if(veg == "Yes"):
        df = df[df['isVegOnly'] == 1]
    elif(veg=="No"):
        df = df[df['isVegOnly']==0]
    else:
        df = df
    if choice2 == "Seating":
        df_filtered = df[(df['AverageCost'] <= budget) & (df['isIndoorSeating'] == 1) & (df['Dinner Ratings'] >= min_rating)]
    elif choice2 == "Order":
        df_filtered = df[(df['AverageCost'] <= budget) & (df['IsHomeDelivery'] == 1) & (df['Delivery Ratings'] >= min_rating)]
    else:
        df_filtered = df[(df['AverageCost'] <= budget) & (df['Dinner Ratings'] >= min_rating)]
    print(df_filtered)
    if df_filtered.empty:
        
        return "None"
    
   
    df_filtered['Cuisine'] = df_filtered['Cuisines'].apply(lambda x: ' '.join(cuisine.lower() for cuisine in str(x).split(', ')))

    
   
    df_filtered['Cuisine'] = df_filtered['Cuisine'].str.lower()


    df_filtered = df_filtered[df_filtered['Cuisine'].apply(lambda x: any(cuisine.lower() in x for cuisine in preferred_cuisines))]

    
   
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df_filtered['Cuisine'])
    
    
   
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    
    
    df_filtered = df_filtered.reset_index()
    indices = pd.Series(df_filtered.index, index=df_filtered['Name'])
    
    
    df_filtered['Unique_ID'] = range(1, len(df_filtered) + 1)
    df_filtered['Unique_ID'] = df_filtered['Unique_ID'].astype(int)
    import matplotlib.pyplot as plt


    def get_recommendations(unique_id, cosine_sim=cosine_sim):
    
        idx = int(unique_id - 1)
       
        

    
        sim_scores = cosine_sim[idx]

    
        threshold = np.mean(sim_scores)

    
        scores = sim_scores


        if any(score > threshold for score in sim_scores):

            sim_scores_with_indices = list(enumerate(sim_scores))
            sim_scores_with_indices = sorted(sim_scores_with_indices, key=lambda x: x[1], reverse=True)

            sim_scores_with_indices = sim_scores_with_indices[1:11]

       
            restaurant_indices = [i[0] for i in sim_scores_with_indices]


            return df_filtered[['Name', 'URL', 'Full_Address', 'AverageCost', 'Cuisines']].iloc[restaurant_indices]
        else:

            return None




   
    recommendations = get_recommendations(df_filtered[df_filtered['Cuisine'].apply(lambda x: any(cuisine.lower() in x for cuisine in preferred_cuisines))]['Unique_ID'].iloc[0])
    
    return recommendations

# Streamlit UI
st.set_page_config(page_title="Search Your Similar Restaurants")

title_style = "font-size: 40px; color: pale-blue; font-family: Georgia, serif;"
st.markdown(f'<h1 style="{title_style}">Search Your Similar Restaurants</h1>', unsafe_allow_html=True)

selected_cuisine = st.text_input("Enter Cuisine", help="Enter Cuisine...")

options = [1.0, 2.0, 3.0, 4.0]
rating = st.selectbox("Enter the desired rating:", options)

budget_input = st.number_input("Enter the budget", min_value=100, max_value=500, value=250, step=1)

options2 = ["Yes", "No","Both"]
choice = st.selectbox("Do you want Veg:", options2)

options3 = ["Seating","Order"]
choice2 = st.selectbox("Select an option:", options3)

if st.button('Show Restaurants'):
    recommendations = recommend_restaurants(df, [selected_cuisine], budget_input, rating, choice,choice2)

    if recommendations is not None:
        st.subheader(f"Recommended Restaurants for {selected_cuisine}")
        st.table(recommendations)
    else:
        st.write("No recommendations found.")

html_content = """
<style>
    *,
    *:after {
      box-sizing: border-box;
    }

    h1 {
      font-size: clamp(20px, 15vmin, 20px);
      font-family: sans-serif;
      color: bloack;
      position: relative;
    }

    h1:after {
      content: "";
      position: absolute;
      width: 100%;
      height: 5px;
      background: hsl(130 80% 50%);
      left: 0;
      bottom: 0;
    }
</style>
"""

st.write(html_content, unsafe_allow_html=True)
