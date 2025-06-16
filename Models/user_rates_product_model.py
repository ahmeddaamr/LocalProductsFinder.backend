from mongoengine import Document, StringField, FloatField, DateTimeField, connect
from datetime import datetime

# Connect to local MongoDB
connect('myapp', host='localhost', port=27017)

class UserRatesProduct(Document):
    user_id = StringField(required=True)
    product_id = StringField(required=True)
    rating = FloatField(required=True, min_value=0.0, max_value=5.0)
    timestamp = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'user_ratings',
        'indexes': [
            {'fields': ['user_id', 'product_id'], 'unique': True}
        ]
    }
