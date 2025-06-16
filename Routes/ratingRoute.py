from flask import Blueprint, request, jsonify
from Controllers.user_rates_product_controller import (
    user_rates_product,
    user_updates_rating,
    user_removes_rating,
    get_all_user_ratings
)
from Middlewares.auth import jwt_required

rating_bp = Blueprint('rating_bp', __name__, url_prefix='/rating')

@rating_bp.route("/add", methods=["POST"]) 
@jwt_required
def add_rating(user_id):
    return user_rates_product(user_id)

@rating_bp.route("/update", methods=["PUT"])
@jwt_required
def update_rating(user_id):
    return user_updates_rating(user_id)

@rating_bp.route("/remove", methods=["DELETE"])
@jwt_required
def remove_rating(user_id):
    return user_removes_rating(user_id)

@rating_bp.route("/getall", methods=["GET"])
@jwt_required
def get_all_ratings(user_id):
    return get_all_user_ratings(user_id)