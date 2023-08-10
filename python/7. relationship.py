# just ignore this
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'pos'))

from pos import http
from pos import config
from pos.models import db

class Config(config.Config):
        SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        
app = http.create_app(Config)
app.app_context().push()
db.create_all()

from pos.models.products import Products
from pos.models.transaction import Transactions
from pos.models.transaction_products import TransactionProducts

# membuat product
food = Products()
food.name = "Makanan"
food.price = 5000
food.stock = 10

beverage = Products()
beverage.name = "Minuman"
beverage.price = 10000
beverage.stock = 10

db.session.add_all([food, beverage])
db.session.flush()

# membuat transaksi
transaction = Transactions()
db.session.add(transaction)
db.session.flush()

# menghubungkan dengan detail transaksi
transaction_products = TransactionProducts()
transaction_products.transaction_id = transaction.id
transaction_products.product_id = food.id
transaction_products.product_qty = 2

transaction_products2 = TransactionProducts()
transaction_products2.transaction_id = transaction.id
transaction_products2.product_id = beverage.id
transaction_products2.product_qty = 2

db.session.add_all([transaction_products, transaction_products2])
db.session.commit()

# mengambil semua product di transaksi
# perhatikan backref pada relationship di model, dengan menggunakan backref('transaction_products') berarti 
# kita bisa membuat model transaction mengakses `TransactionProducts` dengan keyword `transaction_products`
transaction.transaction_products



