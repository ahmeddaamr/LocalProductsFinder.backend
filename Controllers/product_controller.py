from Models.product_model import Product
from flask import jsonify
def get_product(product_id):
    try:
        # Fetch the product from the database using the product_id
        product = Product.objects(product_id=product_id).first()
        
        if not product:
            return jsonify({"error": "Product not found"}), 404
        
        # Convert the product to JSON format
        product_json = product.to_json()
        
        return jsonify(product_json), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def get_all_products():
    try:
        # Fetch all products from the database
        products = Product.objects()
        
        if not products:
            return jsonify({"message": "No products found"}), 404
        
        # Convert each product to JSON format
        products_json = [product.to_json() for product in products]
        
        return jsonify(products_json), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
