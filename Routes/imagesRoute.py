from flask import Blueprint, abort, send_from_directory
import os
images_bp = Blueprint('images_bp', __name__)

@images_bp.route("/image/<category>/<product_id>", methods=["GET"])
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