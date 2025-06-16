from flask import Blueprint, request, jsonify
from Controllers.user_controller import register_user, login_user

user_bp = Blueprint('user_bp', __name__ , url_prefix='/user')

@user_bp.route("/register", methods=["POST"])
def register():
    return register_user()

@user_bp.route("/login", methods=["POST"])
def login():
    return login_user()