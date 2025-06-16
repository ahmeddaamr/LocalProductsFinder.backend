from mongoengine import Document, StringField, DateTimeField, EmailField
from datetime import datetime

class User(Document):
    email = EmailField(required=True, unique=True)
    password_hash = StringField(required=True)  # Store securely (e.g. bcrypt hash)
    name = StringField(required=False)
    joined_at = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'users',
        'indexes': ['email'],
    }

    def to_json(self):
        return {
            "id": str(self.id),
            "email": self.email,
            "name": self.name,
            "joined_at": self.joined_at.isoformat()
        }
