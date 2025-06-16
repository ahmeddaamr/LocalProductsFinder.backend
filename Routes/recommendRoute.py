from flask import Blueprint, request, jsonify
from Services.Recommendation.recommend1 import recommend

recommend_bp = Blueprint('recommend_bp', __name__)

@recommend_bp.route("/recommend/<product_id>", methods=["GET"])
def get_recommendations(product_id):
    # data = request.get_json()
    product_id = int(product_id)
    print("type of product id " , type(product_id))
    if "product_id" == '':
        return jsonify({"error": "Missing product_id"}), 400

    # product_id = data["product_id"]
    recommendations = recommend(product_id)
    # print (recommendations)
    return recommendations, 200