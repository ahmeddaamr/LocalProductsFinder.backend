from flask import request, jsonify
from datetime import datetime
from Models.user_rates_product_model import UserRatesProduct
from Middlewares.auth import jwt_required

# @jwt_required
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
                "timestamp": r.timestamp
            })

        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# # @jwt_required
# def user_rates_product(user_id):
#     try:
#         data = request.get_json()
#         product_id = data["product_id"]
#         rating = float(data["rating"])

#         existing = UserRatesProduct.objects(user_id=user_id, product_id=product_id).first()
#         if existing:
#             return jsonify({"message": "Rating already exists, use update instead."}), 409

#         new_rating = UserRatesProduct(
#             user_id=user_id,
#             product_id=product_id,
#             rating=rating,
#             timestamp=datetime.utcnow()
#         )
#         new_rating.save()

#         return jsonify({"message": "Rating added successfully."}), 201
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# # --- Update an existing rating ---
# # @jwt_required
# def user_updates_rating(user_id):
#     try:
#         data = request.get_json()
#         product_id = data["product_id"]
#         new_rating = float(data["rating"])

#         updated = UserRatesProduct.objects(user_id=user_id, product_id=product_id).modify(
#             set__rating=new_rating,
#             set__timestamp=datetime.utcnow(),
#             new=True
#         )

#         if updated:
#             return jsonify({"message": "Rating updated successfully."}), 200
#         else:
#             return jsonify({"error": "Rating not found."}), 404

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# # --- Remove a user rating ---
# # @jwt_required
# def user_removes_rating(user_id):
#     try:
#         data = request.get_json()
#         product_id = data["product_id"]

#         deleted = UserRatesProduct.objects(user_id=user_id, product_id=product_id).delete()

#         if deleted:
#             return jsonify({"message": "Rating deleted successfully."}), 200
#         else:
#             return jsonify({"error": "Rating not found."}), 404

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400@jwt_required
# # @jwt_required
def user_rates_product(user_id):
    try:
        data = request.get_json()
        product_id = data["product_id"]
        rating = float(data["rating"])

        existing = UserRatesProduct.objects(user_id=user_id, product_id=product_id).first()
        if existing:
            return jsonify({"message": "Rating already exists, use update instead."}), 409

        new_rating = UserRatesProduct(
            user_id=user_id,
            product_id=product_id,
            rating=rating,
            timestamp=datetime.utcnow()
        )
        new_rating.save()

        return jsonify({"message": "Rating added successfully."}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Update an existing rating ---
# @jwt_required
def user_updates_rating(user_id):
    try:
        data = request.get_json()
        product_id = data["product_id"]
        new_rating = float(data["rating"])

        updated = UserRatesProduct.objects(user_id=user_id, product_id=product_id).modify(
            set__rating=new_rating,
            set__timestamp=datetime.utcnow(),
            new=True
        )

        if updated:
            return jsonify({"message": "Rating updated successfully."}), 200
        else:
            return jsonify({"error": "Rating not found."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Remove a user rating ---
# @jwt_required
def user_removes_rating(user_id):
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
