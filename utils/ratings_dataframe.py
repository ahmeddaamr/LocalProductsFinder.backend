import pandas as pd
from Models.user_rates_product_model import UserRatesProduct

# Fetching All Ratings as a DataFrame To use in Collaborative Filtering
def get_all_ratings_dataframe():
    """
    Fetches all user-product ratings from the database and returns a Pandas DataFrame.
    """
    ratings = UserRatesProduct.objects()

    data = []
    for r in ratings:
        data.append({
            "user_id": r.user_id,
            "product_id": r.product_id,
            "rating": r.rating,
            "timestamp": r.timestamp
        })

    df = pd.DataFrame(data)
    return df
