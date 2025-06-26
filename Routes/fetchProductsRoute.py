from flask import Blueprint
from Controllers.fetch_products import fetchProducts
from Controllers.user_rates_product_controller import get_avg_ratings
from utils.merging_products_ratings import merge_products_with_ratings

fetchProducts_bp = Blueprint('fetchProducts_bp', __name__)

@fetchProducts_bp.route("/products", methods=["GET"])
def fetch_products():
    products = fetchProducts()
    # print(products)
    ratings = get_avg_ratings()
    # print(ratings)
    return merge_products_with_ratings(products, ratings), 200