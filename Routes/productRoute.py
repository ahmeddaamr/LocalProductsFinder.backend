from flask import Blueprint, request, jsonify
from Controllers.product_controller import get_product,get_all_products #, update_product

product_bp = Blueprint('product_bp', __name__, url_prefix='/product')

@product_bp.route("add/<product_id>", methods=["GET"])
def getProduct(product_id):
    return get_product(product_id)

@product_bp.route("getll", methods=["GET"])
def getAllProducts():
    return get_all_products()
# @product_bp.route("update/<product_id>", methods=["PUT"])
# def updateProduct(product_id):
#     return update_product(product_id)

