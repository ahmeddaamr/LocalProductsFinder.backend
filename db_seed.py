from Models.product_model import Product
from utils.merging_products_ratings import merge_products_with_ratings
from Controllers.fetch_products import fetchProducts
from Controllers.user_rates_product_controller import get_avg_ratings
from mongoengine import connect
import numpy as np


def seed_db():
    # Ensure DB connection (optional here if done elsewhere)
    print("Connecting to MongoDB...")
    connect('LocalProductsFinder', host='localhost', port=27017)
    print("Connected to MongoDB.")
    # 1️⃣ Fetch products + ratings
    products = fetchProducts()
    print(f"Total products fetched: {len(products)}")
    ratings = get_avg_ratings()
    print(f"Total products fetched: {len(products)}")
    # 2️⃣ Merge ratings into products
    merged_products = merge_products_with_ratings(products, ratings)
    print(f"Total products to upsert: {len(merged_products)}")
    # 3️⃣ Insert or update products in MongoDB
    for p in merged_products:
        try:
            # Upsert: insert or update if exists
            # Product.objects(product_id=int(p["Product ID"])).update_one(
            #     set__product_description=p["Product Description"],
            #     set__product_category=p["Product Category"],
            #     set__sub_category=p["Sub-Category"],
            #     set__local_features=p["Local"] if p["Local"] is not np.nan else "",
            #     set__country=p["Country"] if p["Country"] is not np.nan else "",
            #     set__average_rating=p["average_rating"] if p["average_rating"] is not None else 0.0,
            #     set__ratings_count=p["rating_count"]
            # )
            product = Product.objects(product_id=int(p["Product ID"])).first()
            if not product:
                product = Product(product_id=int(p["Product ID"]))
            product.product_description = p.get("Product Description")
            product.product_category = p.get("Product Category")
            product.sub_category = p.get("Sub-Category")
            product.local_features = p.get("Local features") or ""
            product.country = p.get("Country") or ""
            product.average_rating = p.get("average_rating") if p.get("average_rating") is not None else 0.0
            product.ratings_count = p.get("rating_count", 0)
            product.save()
            print(f"Upserted Product ID {p['Product ID']}")
        except Exception as e:
            print(f"Error upserting Product ID {p['Product ID']}: {e}")

if __name__ == "__main__":
    seed_db()
    print("Database seeding completed.")