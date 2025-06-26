from flask import request, jsonify, current_app
from Models.user_model import User
from datetime import datetime, timedelta
import bcrypt
import jwt

# --- Helper functions for password security ---
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode())

# --- Generate JWT ---
def generate_token(user_id):
    payload = {
        "user_id": str(user_id),
        # "exp": datetime.utcnow() + timedelta(days=7)  # token expires in 7 days
    }
    token = jwt.encode(payload, current_app.config["JWT_SECRET_KEY"], algorithm="HS256")
    return token

# --- Decode JWT ---
def decode_token(token):
    try:
        payload = jwt.decode(token, current_app.config["JWT_SECRET_KEY"], algorithms=["HS256"])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# --- Register User ---
def register_user():
    try:
        data = request.get_json()
        username = data.get('username', '')
        email = data['email']
        password = data['password']
        country = data.get('country', 'Unknown')

        # Check if user already exists
        if User.objects(email=email).first():
            return jsonify({"error": "User already exists"}), 400

        hashed_password = hash_password(password)
        user = User(username=username , email=email, password_hash=hashed_password, country=country)
        user.save()
        token = generate_token(user.id)
        return jsonify({
            "message": "User registered successfully", "user": user.to_json(),
            'token':token}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Login User ---
def login_user():
    try:
        data = request.get_json()
        email = data['email']
        password = data['password']

        user = User.objects(email=email).first()
        if not user or not check_password(password, user.password_hash):
            return jsonify({"error": "Invalid credentials"}), 401

        token = generate_token(user.id)
        return jsonify({ "message": "Login successful",
            "token": token,
            "user": user.to_json()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
