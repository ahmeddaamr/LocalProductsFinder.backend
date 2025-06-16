from flask import Blueprint
from Controllers.fetch_products import fetchProducts

fetchProducts_bp = Blueprint('fetchProducts_bp', __name__)

@fetchProducts_bp.route("/products", methods=["GET"])
def fetch_products():
    return fetchProducts()