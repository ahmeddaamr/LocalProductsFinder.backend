from flask import request, jsonify
from datetime import datetime
from Models.user_rates_product_model import UserRatesProduct
from Models.product_model import Product
from mongoengine.connection import get_db

# --- Get all ratings for a specific user ---
def get_all_user_ratings(user_id):
    try:
        ratings = UserRatesProduct.objects(user_id=user_id)

        if not ratings:
            return jsonify({"message": "No ratings found for this user."}), 404

        data = []
        for r in ratings:
            data.append({
                "product_id": r.product_id,
                "rating": r.rating,
                "review": r.review,
                "timestamp": r.timestamp
            })

        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- User Adds a rating for a specific product ---
def user_rates_product(user_id, product_id):
    try:
        data = request.get_json()
        # product_id = data["product_id"]
        rating = float(data["rating"])
        review = data.get("review", "")
        
        # Check if the product exists
        existing = UserRatesProduct.objects(user_id=user_id, product_id=product_id).first()
        if existing:
            return jsonify({"message": "Rating already exists, use update instead."}), 409
        
        #Add the rating to userRatesProduct collection
        new_rating = UserRatesProduct(
            user_id=user_id,
            product_id=product_id,
            rating=rating,
            review=review,
            timestamp=datetime.utcnow()
        )
        new_rating.save()

        # Update the product's average rating and ratings count
        product = Product.objects(product_id=product_id).first()
        if not product:
            return jsonify({"error": "Product not found."}), 404
        product.ratings_count += 1
        product.average_rating=get_avg_ratings()
        product.save()
        
        return jsonify({"message": "Rating added successfully."}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- User Updates his existing rating ---
def user_updates_rating(user_id , product_id):
    try:
        data = request.get_json()
        # product_id = data["product_id"]
        new_rating = float(data["rating"])
        review = data.get("review", "")

        updated = UserRatesProduct.objects(user_id=user_id, product_id=product_id).modify(
            set__rating=new_rating,
            set__review=review,
            set__timestamp=datetime.utcnow(),
            new=True
        )

        if updated:
            return jsonify({"message": "Rating updated successfully."}), 200
        else:
            return jsonify({"error": "Rating not found."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- User Remove his rating ---
def user_removes_rating(user_id , product_id):
    try:
        data = request.get_json()
        product_id = data["product_id"]

        deleted = UserRatesProduct.objects(user_id=user_id, product_id=product_id).delete()

        if deleted:
            return jsonify({"message": "Rating deleted successfully."}), 200
        else:
            return jsonify({"error": "Rating not found."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Get Ratings for a specific product --- **For Product Page**
def get_product_ratings(product_id):
    try:
        ratings = UserRatesProduct.objects(product_id=product_id)
        if not ratings:
            return jsonify({"message": "No ratings found for this product."}), 404

        # total_rating = sum(r.rating for r in ratings)
        # average_rating = total_rating / len(ratings)

        data=[]
        for r in ratings:
            data.append({
                "user_id": r.user_id,
                "rating": r.rating,
                "review": r.review,
                "timestamp": r.timestamp
            })

        return jsonify({"product_id": product_id, "ratings": data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@staticmethod
def get_avg_ratings():
    db = get_db()
    return list(db.user_ratings.aggregate([
        {
            "$group": {
                "_id": "$product_id",
                "average_rating": {"$avg": "$rating"},
                "rating_count": {"$sum": 1}
            }
        }
    ]))