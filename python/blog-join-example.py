from datetime import datetime

from sqlalchemy import (Column, Integer, Numeric, String, DateTime, 
                        ForeignKey, Boolean, create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref


Base = declarative_base()


class Cookie(Base):
    __tablename__ = 'cookies'

    cookie_id = Column(Integer, primary_key=True)
    cookie_name = Column(String(50), index=True)
    cookie_recipe_url = Column(String(255))
    cookie_sku = Column(String(55))
    quantity = Column(Integer())
    unit_cost = Column(Numeric(12, 2))
    
    def __repr__(self):
        return "Cookie(cookie_name='{self.cookie_name}', "                        "cookie_recipe_url='{self.cookie_recipe_url}', "                        "cookie_sku='{self.cookie_sku}', "                        "quantity={self.quantity}, "                        "unit_cost={self.unit_cost})".format(self=self)
    
    
class User(Base):
    __tablename__ = 'users'
    
    user_id = Column(Integer(), primary_key=True)
    username = Column(String(15), nullable=False, unique=True)
    email_address = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=False)
    password = Column(String(25), nullable=False)
    created_on = Column(DateTime(), default=datetime.now)
    updated_on = Column(DateTime(), default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return "User(username='{self.username}', "                      "email_address='{self.email_address}', "                      "phone='{self.phone}', "                      "password='{self.password}')".format(self=self)
    

class Order(Base):
    __tablename__ = 'orders'
    order_id = Column(Integer(), primary_key=True)
    user_id = Column(Integer(), ForeignKey('users.user_id'))
    shipped = Column(Boolean(), default=False)
    
    user =  relationship("User", backref=backref('orders', order_by=order_id))
    
    def __repr__(self):
        return "Order(user_id={self.user_id}, "                       "shipped={self.shipped})".format(self=self)


class LineItem(Base):
    __tablename__ = 'line_items'
    line_item_id = Column(Integer(), primary_key=True)
    order_id = Column(Integer(), ForeignKey('orders.order_id'))
    cookie_id = Column(Integer(), ForeignKey('cookies.cookie_id'))
    quantity = Column(Integer())
    extended_cost = Column(Numeric(12, 2))
    
    order = relationship("Order", backref=backref('line_items', order_by=line_item_id))
    cookie = relationship("Cookie", uselist=False)

    def __repr__(self):
        return "LineItems(order_id={self.order_id}, "                           "cookie_id={self.cookie_id}, "                           "quantity={self.quantity}, "                           "extended_cost={self.extended_cost})".format(
                    self=self)    


# Connect do an in memory SQLite database                
engine = create_engine('sqlite:///:memory:')
# Create our tables in that database
Base.metadata.create_all(engine)

# Configure our sessions to use the SQLite database engine
Session = sessionmaker(bind=engine)
# Initialize a session
session = Session()

query = session.query(Order.order_id, User.username, User.phone,
                      Cookie.cookie_name, LineItem.quantity,
                      LineItem.extended_cost)
query = query.join(User).join(LineItem).join(Cookie)
results = query.filter(User.username == 'cookiemon').all()

query = session.query(Order.order_id, User.username, User.phone,
                      Cookie.cookie_name, LineItem.quantity,
                      LineItem.extended_cost)
print(query)

query = query.join(User).join(LineItem).join(Cookie)
print(query)

