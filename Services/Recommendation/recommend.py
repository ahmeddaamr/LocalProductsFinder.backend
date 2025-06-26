import pandas as pd
import numpy as np
import string
import os
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lightfm.data import Dataset
from lightfm import LightFM
from sentence_transformers import SentenceTransformer
import nltk
import re
from collections import Counter

# Download necessary NLTK resources (uncomment if needed)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('punkt_tab')

def preprocess_text(text):
    """Text preprocessing matching recommend1.py approach"""
    if pd.isna(text):
        return ""
    
    lemmatizer = WordNetLemmatizer()
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_tokens)

def create_combined_features(df):
    """Create combined features similar to recommend1.py"""
    # Add Local_Status
    df['Local_Status'] = np.where(df['Local'] == 'Yes', 'local', 'not_local')
    
    # Create Combined_Features like in recommend1.py
    df["Combined_Features"] = (
        df["Product Description"] + " " +
        df["Product Category"] + " " +
        df["Sub-Category"]
    ).apply(preprocess_text)
    
    # Apply weighting similar to recommend1.py
    df["Combined_Features"] = df.apply(lambda x:
        (x["Local_Status"] + " ") * 3 +
        (x["Product Category"] + " ") * 3 +  # Increase category weight
        (x["Sub-Category"] + " ") * 3 +  # Increase sub-category weight
        x["Combined_Features"], axis=1)
    
    return df

def compute_enhanced_content_similarity(input_product_id, candidate_products_df, 
                                      all_product_embeddings_dict, 
                                      all_products_metadata_df,
                                      model="bert",
                                      vectorizer=None,
                                      tfidf_matrix=None):
    """
    Enhanced content similarity computation using recommend1.py approach
    """
    print(f"🔄 Computing content similarity for product {input_product_id} using {model}")
    
    # Get input product details
    input_row = all_products_metadata_df[
        all_products_metadata_df['Product ID'] == input_product_id
    ]
    if input_row.empty:
        print(f"❌ Input product {input_product_id} not found")
        return []
    
    input_row = input_row.iloc[0]
    input_combined = preprocess_text(input_row["Combined_Features"])
    
    # Compute similarities based on model type
    similarities = []
    
    if model == "tfidf" and vectorizer is not None:
        # TF-IDF approach from recommend1.py
        input_vector = vectorizer.transform([input_combined])
        
        for _, candidate_row in candidate_products_df.iterrows():
            candidate_id = str(candidate_row['Product ID'])
            
            # Skip self-comparison
            if candidate_id == input_product_id:
                continue
            
            candidate_combined = preprocess_text(candidate_row["Combined_Features"])
            candidate_vector = vectorizer.transform([candidate_combined])
            similarity = cosine_similarity(input_vector, candidate_vector)[0][0]
            
            similarities.append({
                'id': candidate_id,
                'similarity': similarity,
                'description': candidate_row['Product Description'],
                'category': candidate_row['Product Category'],
                'subcategory': candidate_row['Sub-Category']
            })
    
    elif model == "bert":
        # BERT approach from recommend1.py
        if input_product_id not in all_product_embeddings_dict:
            print(f"❌ Embedding not found for input product {input_product_id}")
            return []
        
        input_embedding = all_product_embeddings_dict[input_product_id].reshape(1, -1)
        
        for _, candidate_row in candidate_products_df.iterrows():
            candidate_id = str(candidate_row['Product ID'])
            
            # Skip self-comparison
            if candidate_id == input_product_id:
                continue
                
            # Skip if embedding not available
            if candidate_id not in all_product_embeddings_dict:
                continue
            
            candidate_embedding = all_product_embeddings_dict[candidate_id].reshape(1, -1)
            similarity = cosine_similarity(input_embedding, candidate_embedding)[0][0]
            
            similarities.append({
                'id': candidate_id,
                'similarity': similarity,
                'description': candidate_row['Product Description'],
                'category': candidate_row['Product Category'],
                'subcategory': candidate_row['Sub-Category']
            })
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"✅ Computed similarities for {len(similarities)} candidates")
    
    # Debug: Show top similarities
    print("🔍 Top 5 content similarities:")
    for i, sim in enumerate(similarities[:5], 1):
        print(f"  {i}. {sim['description'][:50]}...")
        print(f"     Similarity: {sim['similarity']:.4f}")
    
    return similarities

def generate_recommendations(user_id: str, product_id: str, df_interaction: pd.DataFrame) -> list:
    # Load product data
    if not os.path.exists("/Dataset/Product_Final.xlsx"):
        print("Product dataset not found.")
        return []
    
    df_products = pd.read_excel("Data/Product_Final.xlsx")

    # Clean and prepare product data (keeping existing cleaning)
    df_products['Product ID'] = df_products['Product ID'].astype(str)
    df_products['Product Category'] = df_products['Product Category'].astype(str).str.strip().str.lower()
    df_products['Sub-Category'] = df_products['Sub-Category'].astype(str).str.strip().str.lower()
    df_products['Product Description'] = df_products['Product Description'].astype(str).str.strip()
    df_products['Local'] = df_products['Local'].astype(str).str.strip().str.lower()
    df_products['features'] = df_products['features'].astype(str).fillna('')
    
    # Create combined features like recommend1.py
    df_products = create_combined_features(df_products)
    df_products_local = df_products[df_products['Local'] == 'yes'].copy()

    # Check if user_id is None - if so, skip collaborative filtering entirely
    if user_id is None:
        print("🔄 User ID is None - using content-based recommendations only")
        has_interaction_data = False
        lightfm_model = None
        user_id_map = {}
        item_id_map = {}
        df_interaction = pd.DataFrame()  # Empty interaction data
    else:
        # Check if we have any interaction data
        has_interaction_data = len(df_interaction) > 0 and not df_interaction.empty
        
        # Initialize CF components only if we have interaction data
        lightfm_model = None
        user_id_map = {}
        item_id_map = {}
        
        if has_interaction_data:
            print("✅ Found interaction data. Setting up collaborative filtering...")
            
            # Clean interaction data - now using the new structure
            df_interaction['user_id'] = df_interaction['user_id'].astype(str).str.strip()
            df_interaction['product_id'] = df_interaction['product_id'].astype(str).str.strip()
            df_interaction['rating'] = pd.to_numeric(df_interaction['rating'], errors='coerce')

            # Remove rows with invalid ratings
            df_interaction = df_interaction.dropna(subset=['rating'])
            df_interaction = df_interaction[df_interaction['rating'] > 0]

            # Parse timestamp and compute recency weights
            df_interaction['timestamp'] = pd.to_datetime(df_interaction['timestamp'], errors='coerce')
            df_interaction = df_interaction.dropna(subset=['timestamp'])

            max_time = df_interaction['timestamp'].max()
            min_time = df_interaction['timestamp'].min()
            time_range = (max_time - min_time).total_seconds()

            def compute_recency_weight(ts):
                delta = (ts - min_time).total_seconds()
                return 0.1 + 0.9 * (delta / time_range) if time_range > 0 else 1.0

            df_interaction['recency_weight'] = df_interaction['timestamp'].apply(compute_recency_weight)

            # Use product_id directly
            df_interaction['ProductID'] = df_interaction['product_id']
            df_interaction['ProductID'] = df_interaction['ProductID'].astype(str)

            # Only proceed with LightFM if we have valid interaction data after mapping
            if len(df_interaction) > 0:
                # Rebuild LightFM data structures
                unique_users = sorted(df_interaction['user_id'].unique())
                unique_items = sorted(df_interaction['ProductID'].unique())

                dataset = Dataset()
                dataset.fit(users=unique_users, items=unique_items)

                interaction_tuples = [
                    (str(row['user_id']), str(row['ProductID']), float(row['rating']) * float(row['recency_weight']))
                    for _, row in df_interaction.iterrows()
                ]

                if interaction_tuples:  # Only build model if we have valid interactions
                    interactions, _ = dataset.build_interactions(interaction_tuples)
                    lightfm_model = LightFM(loss='bpr', no_components=50, random_state=42,
                                          item_alpha=1e-6, user_alpha=1e-6)
                    lightfm_model.fit(interactions, epochs=20, num_threads=4)

                    # Rebuild mappings
                    user_id_map, _, item_id_map, _ = dataset.mapping()
                    print(f"✅ Trained CF model with {len(unique_users)} users and {len(unique_items)} items")
                else:
                    print("⚠️ No valid interaction tuples found after processing")
                    has_interaction_data = False
            else:
                print("⚠️ No valid interactions after mapping product names to IDs")
                has_interaction_data = False
        else:
            print("⚠️ No interaction data found. Using content-based recommendations only.")

    # Initialize Vectorizers like recommend1.py
    print("🔄 Setting up vectorizers and embeddings...")
    vectorizer = TfidfVectorizer()
    bert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # Fit TF-IDF Model on combined features
    tfidf_matrix = vectorizer.fit_transform(df_products["Combined_Features"])
    
    # Generate BERT Embeddings
    df_products["BERT_Embedding"] = list(bert_model.encode(df_products["Combined_Features"].tolist()))
    
    # Create embeddings dictionary for compatibility
    product_embeddings = {
        str(row['Product ID']): df_products.iloc[i]["BERT_Embedding"]
        for i, (_, row) in enumerate(df_products.iterrows())
    }
    print(f"✅ Created embeddings for {len(product_embeddings)} products")

    # Call enhanced hybrid recommender function
    recommendations = recommend_top5_hybrid_enhanced(
        user_id=user_id,
        product_input=product_id,
        df_all_products_metadata=df_products,
        df_local_products_metadata=df_products_local,
        df_interaction_data=df_interaction if has_interaction_data else pd.DataFrame(),
        user_id_to_idx_lightfm=user_id_map,
        item_id_to_idx_lightfm=item_id_map,
        all_product_embeddings_dict=product_embeddings,
        lightfm_model=lightfm_model,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix
    )

    return recommendations


def recommend_top5_hybrid_enhanced(user_id, 
                                 product_input, 
                                 df_all_products_metadata,
                                 df_local_products_metadata,
                                 df_interaction_data,
                                 user_id_to_idx_lightfm,
                                 item_id_to_idx_lightfm,
                                 all_product_embeddings_dict,
                                 lightfm_model,
                                 vectorizer=None,
                                 tfidf_matrix=None,
                                 top_k_content_candidates=50):
    """
    Enhanced hybrid recommender with content-based approach matching recommend1.py
    """
    
    user_id_str = str(user_id) if user_id is not None else None
    print(f"🔄 Generating recommendations for user: {user_id_str}")
    print(f"🔄 Product input: {product_input}")

    # 1. Resolve product ID from input
    product_id_input_str = None
    if isinstance(product_input, (int, str)) and str(product_input).isdigit():
        if str(product_input) in df_all_products_metadata['Product ID'].astype(str).values:
            product_id_input_str = str(product_input)
    else:
        desc_clean = str(product_input).strip().lower()
        matched_row = df_all_products_metadata[
            df_all_products_metadata['Product Description'].str.lower() == desc_clean
        ]
        if not matched_row.empty:
            product_id_input_str = str(matched_row.iloc[0]['Product ID'])
        else:
            # Try partial matching if exact match fails
            partial_matches = df_all_products_metadata[
                df_all_products_metadata['Product Description'].str.lower().str.contains(desc_clean, na=False)
            ]
            if not partial_matches.empty:
                product_id_input_str = str(partial_matches.iloc[0]['Product ID'])
                print(f"🟡 Used partial match for product: {partial_matches.iloc[0]['Product Description']}")

    if not product_id_input_str:
        print(f"❌ Could not resolve product input '{product_input}' to a Product ID.")
        return []

    print(f"✅ Resolved product ID: {product_id_input_str}")

    # Check if user exists in CF model (cold start detection) - skip if user_id is None
    use_cf = (user_id is not None and 
              lightfm_model is not None and 
              user_id_str in user_id_to_idx_lightfm and 
              len(df_interaction_data) > 0)
    
    if not use_cf:
        if user_id is None:
            print(f"🟡 No user ID provided — using content-based only.")
        else:
            print(f"🟡 Cold-start user '{user_id_str}' or no CF model — using content-based only.")

    # Check if product embedding exists
    if product_id_input_str not in all_product_embeddings_dict:
        print(f"❌ Embedding not found for input product ID '{product_id_input_str}'.")
        return []

    # 2. Get input product info and apply local boost like recommend1.py
    scanned_row = df_all_products_metadata[df_all_products_metadata['Product ID'] == product_id_input_str]
    if scanned_row.empty:
        print(f"❌ Scanned product '{product_id_input_str}' not found in metadata.")
        return []

    input_product = scanned_row.iloc[0]
    is_local = input_product["Local"] == "yes"
    
    # Filter out the input product from candidates
    filtered_df = df_local_products_metadata[df_local_products_metadata["Product ID"] != product_id_input_str].copy()
    print(f"🔄 Found {len(filtered_df)} local products (excluding input product)")

    # 3. Content similarity computation using BERT (like recommend1.py)
    similarity_results = compute_enhanced_content_similarity(
        input_product_id=product_id_input_str,
        candidate_products_df=filtered_df,
        all_product_embeddings_dict=all_product_embeddings_dict,
        all_products_metadata_df=df_all_products_metadata,
        model="bert",
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix
    )

    if not similarity_results:
        print("❌ No similar products found")
        return []

    # Apply local boost like recommend1.py
    LOCAL_BOOST = 0.9
    for item in similarity_results:
        candidate_row = df_local_products_metadata[df_local_products_metadata['Product ID'] == item['id']]
        if not candidate_row.empty and candidate_row.iloc[0]['Local'] == 'yes':
            item['boosted_similarity'] = item['similarity'] * LOCAL_BOOST
        else:
            item['boosted_similarity'] = item['similarity']

    # Sort by boosted similarity
    similarity_results.sort(key=lambda x: x['boosted_similarity'], reverse=True)

    # Extract top candidates for CF scoring
    top_k_ids = [item['id'] for item in similarity_results[:top_k_content_candidates]]
    
    # For CF, we need items that exist in the CF model
    if use_cf:
        available_cf_ids = [item_id for item_id in top_k_ids if item_id in item_id_to_idx_lightfm]
        missing_cf_ids = [item_id for item_id in top_k_ids if item_id not in item_id_to_idx_lightfm]
        
        if not available_cf_ids:
            print("⚠️ None of the candidates are in CF model — will fall back to content-based scores only.")
        else:
            print(f"✅ {len(available_cf_ids)} candidates found in CF model, {len(missing_cf_ids)} missing.")
    else:
        available_cf_ids = []
        missing_cf_ids = top_k_ids

    item_interaction_counts = {}
    if use_cf and 'product_id' in df_interaction_data.columns:
        item_interaction_counts = df_interaction_data['product_id'].value_counts().to_dict()

    print(f"✅ Selected {len(top_k_ids)} candidates for final scoring")

    # 4. Compute scores using boosted similarity (like recommend1.py)
    content_scores_map = {item['id']: item['boosted_similarity'] for item in similarity_results 
                         if item['id'] in top_k_ids}

    cf_scores_map = {}
    if use_cf:
        try:
            user_idx = user_id_to_idx_lightfm[user_id_str]
            item_indices = [item_id_to_idx_lightfm[i] for i in top_k_ids if i in item_id_to_idx_lightfm]
            available_item_ids = [i for i in top_k_ids if i in item_id_to_idx_lightfm]
            cf_scores_raw = lightfm_model.predict(user_ids=user_idx, item_ids=np.array(item_indices))
            cf_scores_map = dict(zip(available_item_ids, cf_scores_raw))
            print(f"✅ Computed CF scores for {len(cf_scores_map)} items")
        except Exception as e:
            print(f"⚠️ Error computing CF scores: {e}")
            use_cf = False

    # 5. Adaptive alpha (weight between CF and content-based)
    if use_cf:
        user_interactions = df_interaction_data[df_interaction_data['user_id'] == user_id_str]
        num_pos_interactions = len(user_interactions[user_interactions['rating'] >= 4])
        if num_pos_interactions > 10:
            alpha = min(0.9, max(0.1, num_pos_interactions / 100))
        else:
            alpha = 0.1  # Small CF weight for users with few interactions
        print(f"✅ Using alpha = {alpha:.2f} (CF weight) based on {num_pos_interactions} positive interactions")
    else:
        alpha = 0.0  # full fallback to content-based
        print("✅ Using alpha = 0.0 (content-based only)")

    hybrid_scored_items = []
    k = 5  # Smoothing factor for item gating

    for item_id in top_k_ids:
        cf_s = max(0.0, cf_scores_map.get(item_id, 0.0))  # clamp negative scores
        cb_s = content_scores_map.get(item_id, 0.0)

        # Compute soft gating based on item interaction count
        item_rating_count = item_interaction_counts.get(item_id, 0)
        g_i = item_rating_count / (item_rating_count + k)

        # Combine using both alpha (user trust) and g_i (item trust)
        final_alpha = alpha * g_i
        hybrid_score = final_alpha * cf_s + (1 - final_alpha) * cb_s

        hybrid_scored_items.append({
            'id': item_id,
            'score': hybrid_score,
            'cf': cf_s,
            'cb': cb_s
        })

    hybrid_scored_items = sorted(hybrid_scored_items, key=lambda x: x['score'], reverse=True)[:5]

    # 6. Format results
    results = []
    for item in hybrid_scored_items:
        row = df_local_products_metadata[df_local_products_metadata['Product ID'] == item['id']]
        if row.empty:
            continue
        row = row.iloc[0]
        
        # Find the similarity details for this item
        sim_details = next((s for s in similarity_results if s['id'] == item['id']), {})
        
        results.append({
            'Product ID': item['id'],
            'Description': row['Product Description'],
            'Category': row['Product Category'],
            'Sub-Category': row['Sub-Category'],
            'Local': row['Local'],
            'Score': round(item['score'], 4),
            'CF_Score_Debug': round(item['cf'], 4),
            'CB_Score_Debug': round(item['cb'], 4),
            'Boosted_Similarity': round(sim_details.get('boosted_similarity', 0), 4)
        })

    print(f"✅ Generated {len(results)} recommendations")
    return results


def recommend(product_id, df_interaction, user_id = None):
    if user_id is None:
        print("🔄 User ID is None - using content-based recommendations only")
        recommendations = generate_recommendations(user_id, product_id, df_interaction)
        print("Done content-based recommendations")
        return recommendations
    else:
        print(" ✅ Found User ID")
        recommendations = generate_recommendations(user_id, product_id, df_interaction)
        print("Done hybrid recommendations")
        return recommendations


if __name__ == "__main__":
    df_interaction = pd.read_csv("Data/interaction_history.csv")
    user_id = "12345"
    product_id = 4009
    recommendations = recommend(product_id, df_interaction, user_id)
    print("="*80)
    for rec in recommendations:
        print(f"Product ID: {rec['Product ID']}, Score: {rec['Score']}, Description: {rec['Description'][:50]}")