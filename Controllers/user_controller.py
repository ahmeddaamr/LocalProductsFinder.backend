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
        print(data)
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
    
def logout_user(user_id):
    try:
        # data = request.get_json()
        # print(data)

        print(user_id)
        user = User.objects(id=user_id).first()
        print(user.email)

        if user:
            return jsonify({"message": "Logout successful"}) ,204
        else:
            return jsonify({"error":" User Does Not Exist! "}),404
    except Exception as e:
        return jsonify({"error":str(e)}) ,400

def fetch_user(user_id):
    try:
        user = User.objects(id=user_id).first()
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        return jsonify({"user": user.to_json()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def update_user(user_id):
    user = User.objects(id=user_id).first()

    if not user:
        return jsonify({"error": "User does not exist"}), 404

    data = request.get_json()

    user.email = data.get("email", user.email)
    user.username = data.get("username", user.username)
    user.country = data.get("country",user.country)
    old_password = data.get("old_password")
    new_password = data.get("new_password")

    if old_password or new_password:
        if not old_password or not new_password:
            return jsonify({"error": "Both old and new passwords are required"}), 400

        if old_password == new_password:
            return jsonify({"error": "New password has been used before"}), 400

        if check_password(old_password, user.password_hash):
            user.password_hash = hash_password(new_password)
        else:
            return jsonify({"error": "Old password is incorrect"}), 400

    user.save()
    return jsonify({"message": "User details updated successfully"}), 200
