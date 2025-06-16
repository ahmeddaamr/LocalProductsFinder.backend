from flask import Blueprint, request, jsonify , current_app
from Services.Identification.identification import predict
import os

predict_bp = Blueprint('predict_bp', __name__)

@predict_bp.route("/predict", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 401
        
    print("File not saved yet:")
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    print("File saved at:", filepath)
    result = predict(filepath)
    if os.remove(filepath):
        print ("image removed from local storage successfully")

    print(result)

    # recom = get_recommendations(result['product_id'])

    # res={}
    # res['identification']=result
    # res['recommendation']=recom
    return result, 200