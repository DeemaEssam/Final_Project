import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('dataset.csv')

# Define personas
subtype_to_persona = {
    'Religious Sites': 'Spiritual Seeker',
    'Historical Sites & Landmarks': 'History Buff',
    'Nature & Parks': 'Nature Enthusiast',
    'Museums': 'Art & Culture Lover',
    'Shopping': 'Shopaholic',
    'Tours & Adventures': 'Adventurer',
    'Restaurants (Food & Dining)': 'Foodie',
    'Wellness & Fitness': 'Health Conscious',
    'Miscellaneous': 'Explorer',
    'Entertainment': 'Entertainment Seeker'
}

def get_persona(preferred_subtype):
    """Map user's preferred subtype to a persona."""
    return subtype_to_persona.get(preferred_subtype, "Explorer")

def filter_and_prepare_data(preferred_subtype, preferred_city):
    """Filter dataset based on user preferences (theme and city) and prepare text for TF-IDF."""
    
    if isinstance(df, pd.DataFrame):  # Ensure df is a DataFrame    
        # Check if preferred_subtype is a valid string
        if not isinstance(preferred_subtype, str):
            raise ValueError(f"Expected 'preferred_subtype' to be a string, got {type(preferred_subtype)}")
        
        # Filter by theme and city
        print(f"Filtering data for subtype: {preferred_subtype}, city: {preferred_city}")  # Debug log
        filtered_df = df[df['subtype'].str.lower().str.contains(preferred_subtype.lower(), na=False)]
        filtered_df = filtered_df[filtered_df['city'].str.lower().str.contains(preferred_city.lower(), na=False)]
        
        print(f"Filtered DataFrame shape: {filtered_df.shape}")  # Debug log

        # Combine text for TF-IDF
        filtered_df.loc[:, 'combined_text'] = (
            filtered_df['description_x'].fillna('') + " " + filtered_df['text'].fillna('')
        ).str.strip()

        if filtered_df.empty:
            print("No matching places found for your preferences.")  # Debug log
            return None
        return filtered_df
    else:
        print("Error: df is not a valid DataFrame!")  # Debug log
        return None

def train_tfidf(filtered_df):
    """Train TF-IDF on the filtered dataset."""
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_df['combined_text'])
    return tfidf_vectorizer, tfidf_matrix

def recommend(preferred_subtype, keywords, preferred_city):
    """Recommend top destinations based on user's persona and keywords."""
    filtered_df = filter_and_prepare_data(preferred_subtype, preferred_city)

    if filtered_df is None:
        return []

    tfidf_vectorizer, tfidf_matrix = train_tfidf(filtered_df)
    
    # Convert user keywords to TF-IDF vector
    user_keywords_vector = tfidf_vectorizer.transform([" ".join(keywords)])

    # Compute similarity
    similarities = cosine_similarity(user_keywords_vector, tfidf_matrix).flatten()

    filtered_df['similarity'] = similarities
    recommendations = filtered_df.sort_values(by='similarity', ascending=False).head(50)

    if recommendations.empty:
        print("No recommendations found based on the keywords and preferences.")  # Debug log
    
    return recommendations[['id','name', 'city', 'subtype', 'description_x', 'imageUrl_x', 'placeRating','website']].to_dict(orient='records')
