from flask import Blueprint, request, jsonify
from Controllers.user_controller import logout_user, register_user, login_user , fetch_user ,update_user
from Middlewares.auth import jwt_required

user_bp = Blueprint('user_bp', __name__ , url_prefix='/user')

@user_bp.route("/register", methods=["POST"])
def register():
    return register_user()

@user_bp.route("/login", methods=["POST"])
def login():
    return login_user()

@user_bp.route("/get/<user_id>", methods=["GET"])
def fetchUser(user_id):
    return fetch_user(user_id)

@user_bp.route("/logout", methods=["DELETE"])
@jwt_required
def logout(user_id):
    return logout_user(user_id)

@user_bp.route("/update", methods=["PUT"])
@jwt_required
def updateUser(user_id):
    return update_user(user_id)
