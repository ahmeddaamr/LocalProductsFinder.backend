from flask import Flask, abort, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from Models.Recommendation.recommend1 import recommend
from Models.Identification.test import predict
from Dataset.fetch_products import fetchProducts

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    return jsonify({"message": "File uploaded", "filename": file.filename}), 200

# @app.route("/image/<filename>", methods=["GET"])
# def get_image(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# @app.route("/images", methods=["GET"])
# def get_all_images():
#     """Fetch all image filenames in the uploads folder"""
#     files = os.listdir(app.config["UPLOAD_FOLDER"])
#     images = [f"http://127.0.0.1:5000/image/{file}" for file in files]  # Generate URLs
#     return jsonify({"images": images})

@app.route("/predict", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 401

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    result = predict(filepath)
    if os.remove(filepath):
        print ("image removed from local storage successfully")

    print(result)

    # recom = get_recommendations(result['product_id'])

    # res={}
    # res['identification']=result
    # res['recommendation']=recom
    return result, 200


@app.route("/recommend/<product_id>", methods=["GET"])
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

@app.route("/image/<category>/<product_id>", methods=["GET"])
def get_image(category, product_id):
    folder_path = f"./Dataset/Images/{category}"  # Path to the category folder
    possible_extensions = ["jpg", "png", "jpeg","avif","webp"]  # Extensions to try
    filename_base = f"{product_id}_1"  # Base filename format

    for ext in possible_extensions:
        filename = f"{filename_base}.{ext}"
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):  # Check if file exists
            return send_from_directory(folder_path, filename)

    # If no file is found, return a 404 error
    abort(404, description="Image not found")

@app.route("/products", methods=["GET"])
def fetch_products():
    return fetchProducts()
    
if __name__ == "__main__":
    app.run(debug=True)
    # app.run(debug=True, host="0.0.0.0", port=5000)
    