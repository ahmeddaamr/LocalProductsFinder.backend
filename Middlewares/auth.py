import jwt
from functools import wraps
from flask import request, jsonify, current_app

def decode_token(token):
    try:
        payload = jwt.decode(token, current_app.config["JWT_SECRET_KEY"], algorithms=["HS256"])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def jwt_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401
        token = auth_header.split(" ")[1]
        user_id = decode_token(token)
        if not user_id:
            return jsonify({"error": "Invalid or expired token"}), 401
        return f(user_id=user_id, *args, **kwargs)
    return decorated_function
