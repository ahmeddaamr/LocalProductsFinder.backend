from mongoengine import Document, IntField, StringField, BooleanField ,FloatField

class Product(Document):
    product_id = IntField(required=True, unique=True)
    product_description = StringField(required=True)
    product_category = StringField(required=True)
    sub_category = StringField(required=True)
    local_features = StringField()
    country = StringField()
    average_rating = FloatField(default=0.0)
    ratings_count = IntField(default=0)

    meta = {'collection': 'products'}
    def to_json(self):
        return {
            "Product ID": self.product_id,
            "Product Description": self.product_description,
            "Product Category": self.product_category,
            "Sub-category": self.sub_category,
            "local": self.local_features,
            "Country": self.country,
            "average_rating": self.average_rating,
            "rating_count": self.ratings_count
        }
    
    @staticmethod
    def getproduct(product_id):
        print(f"Fetching product with ID: {product_id}")
        product = Product.objects(product_id=product_id).first()
        if product:
            product_json = product.to_json()
            print(f"Product found: {product_json}")
            return product_json
        else:
            print(f"Product with ID {product_id} not found.")
            return None