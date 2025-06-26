from flask import Flask, abort, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from mongoengine import connect , disconnect , get_connection
from Routes.fetchProductsRoute import fetchProducts_bp
from Routes.imagesRoute import images_bp
from Routes.predictRoute import predict_bp
from Routes.recommendRoute import recommend_bp
from Routes.userRoute import user_bp
from Routes.ratingRoute import rating_bp
import atexit, signal, sys
# from ngrok_host import ngrok_host 

app = Flask(__name__)

# Enable CORS for cross-origin requests
CORS(app)

#disconnect from MongoDB if already connected
# if(get_connection()):
disconnect()
print("[MongoDB] Disconnected (initialization)")
# Connect to the MongoDB database
connect('LocalProductsFinder', host='localhost', port=27017)
print("[MongoDB] Connected to LocalProductsFinder at : localhost:27017")

# # On process exit (normal termination, including Ctrl+C)
# @atexit.register
# def on_exit():
#     disconnect()
#     print("[MongoDB] Disconnected (atexit)")

# On SIGINT (Ctrl+C)
def handle_sigint(sig, frame):
    disconnect()
    print("\n[MongoDB] Disconnected (SIGINT)")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# # Disconnect on app context teardown
# @app.teardown_appcontext
# def close_mongo_connection(exception=None):
#     disconnect()
#     print("[MongoDB] Disconnected on shutdown.")

# Set the upload folder for images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Set the JWT secret key for authentication
app.config["JWT_SECRET_KEY"] = "your_secret_key_here"  # keep this safe

#command to run ngrok manually
#ngrok http --domain=noble-lemur-humbly.ngrok-free.app 5000

# ngrok_host()  # Start ngrok to expose the app

# Register blueprints
app.register_blueprint(predict_bp)
app.register_blueprint(recommend_bp)
app.register_blueprint(images_bp)
app.register_blueprint(fetchProducts_bp)
app.register_blueprint(user_bp)
app.register_blueprint(rating_bp)

    
if __name__ == "__main__":
    app.run(debug=True)
    # app.run(debug=True, host="0.0.0.0", port=5000)
    