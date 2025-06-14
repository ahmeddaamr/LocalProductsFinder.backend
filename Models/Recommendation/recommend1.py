import pandas as pd
import numpy as np
import string
import os
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import nltk

# Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('punkt_tab')

# Set file path
file_path = "../../Dataset/Product_Final.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Ensure it's in the correct directory.")

# Load Data
df = pd.read_excel(file_path)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_tokens)

df['Local_Status'] = np.where(df['Local'] == 'Yes', 'local', 'not_local')
df["Combined_Features"] = (
        df["Product Description"] + " " +
        df["Product Category"] + " " +
        df["Sub-Category"]
        ).apply(preprocess_text)

df["Combined_Features"] = df.apply(lambda x:
    (x["Local_Status"] + " ") * 3 +
    (x["Product Category"] + " ") * 3 +  # Increase category weight
    (x["Sub-Category"] + " ") * 3 +  # Increase sub-category weight
    x["Combined_Features"], axis=1)

local_products = df[df["Local"] == "Yes"].copy()

# Initialize Vectorizers
vectorizer = TfidfVectorizer()
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
#vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.8, min_df=2)
#vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Using unigrams, bigrams, and trigrams


# Fit TF-IDF Model
tfidf_matrix = vectorizer.fit_transform(df["Combined_Features"])
# Generate BERT Embeddings
df["BERT_Embedding"] = list(bert_model.encode(df["Combined_Features"].tolist()))
# print(df["Combined_Features"])

def get_recommendations(product_id, model="tfidf", top_n=10):
    if product_id not in df["Product ID"].values:
        raise ValueError(f"Product ID {product_id} not found in the dataset.")

    input_product = df[df["Product ID"] == product_id].iloc[0]
    input_combined = preprocess_text(input_product["Combined_Features"])
    is_local = input_product["Local"] == "Yes"

    filtered_df = df[df["Product ID"] != product_id].copy()

    if model == "tfidf":
        input_vector = vectorizer.transform([input_combined])
        similarity_scores = cosine_similarity(input_vector, vectorizer.transform(filtered_df["Combined_Features"])).flatten()

    elif model == "bert":
        input_embedding = bert_model.encode([input_combined])[0]
        filtered_embeddings = np.stack(filtered_df["BERT_Embedding"].values)
        similarity_scores = cosine_similarity([input_embedding], filtered_embeddings).flatten()

    filtered_df["Similarity"] = similarity_scores
    LOCAL_BOOST = 0.9
    filtered_df["Boosted_Similarity"] = filtered_df["Similarity"] * np.where(filtered_df["Local"] == "Yes", LOCAL_BOOST, 1.0)
    
    local_recommendations = filtered_df[filtered_df["Local"] == "Yes"].copy()

    final_recommendations = local_recommendations.sort_values(by="Boosted_Similarity", ascending=False).head(top_n)
    
    # Format the results for JSON output
    result = final_recommendations[["Product ID", "Product Description", "Product Category", "Sub-Category", "Local", "Boosted_Similarity"]].copy()
    
    # Convert float values to more readable format
    result["Boosted_Similarity"] = result["Boosted_Similarity"].round(4)
    
    # Return a list of dictionaries (ideal for JSON serialization)
    return result.to_dict(orient="records")

def recommend(product_id):
    input_product_id = product_id
    try:
        # print("TF-IDF Recommendations:")
        result_tdfidf = get_recommendations(input_product_id, model="tfidf")
        # print(result_tdfidf)
        print("\nBERT Recommendations:")
        result_BERT = get_recommendations(input_product_id, model="bert")
        print(result_BERT)
        
        # Now result_BERT is already in a format suitable for JSON
        return result_BERT
    except ValueError as e:
        print(e)
        return {"error": str(e)}