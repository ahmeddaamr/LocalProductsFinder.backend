import json
import pandas as pd
from Models.product_model import Product


def merge_products_with_ratings(products, ratings):
    
    if isinstance(products, str):
        products = json.loads(products)
    if isinstance(ratings, str):
        ratings = json.loads(ratings)

    rating_map = {r["_id"]: r for r in ratings}
    for product in products:
        r = rating_map.get(product["Product ID"])
        if r:
            product["average_rating"] = round(r["average_rating"], 2)
            product["rating_count"] = r["rating_count"]
        else:
            product["average_rating"] = None
            product["rating_count"] = 0
    return products
