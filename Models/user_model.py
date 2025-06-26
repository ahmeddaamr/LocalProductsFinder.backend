from mongoengine import Document, StringField, DateTimeField, EmailField
from datetime import datetime

class User(Document):
    username = StringField(required=True, unique=True)
    email = EmailField(required=True, unique=True)
    password_hash = StringField(required=True)  # Store securely (e.g. bcrypt hash)
    country = StringField(required=True)
    joined_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'users',
        'indexes': ['email'],
    }

    def to_json(self):
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "country": self.country,
            "joined_at": self.joined_at.isoformat()
        }
