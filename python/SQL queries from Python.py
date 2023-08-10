from sys import version 
import sqlalchemy
import IPython
print ' Reproducibility conditions for this notebook '.center(90,'-')
print 'Python version:     ' + version
print 'SQLalchemy version: ' + sqlalchemy.__version__
print 'IPython version:    ' + IPython.__version__
print '-'*90

from sqlalchemy import MetaData
metadata = MetaData()

from sqlalchemy import Table, Column, Integer, Numeric, String, ForeignKey, DateTime
from datetime import datetime

from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')

users = Table('users', metadata, 
              Column('user_id', Integer(), primary_key=True), # table’s primary key
              Column('username', String(15), nullable=False, unique=True, index = True),
              Column('email_address', String(255), nullable=False),
              Column('phone', String(20), nullable=False),
              Column('password', String(25), nullable=False),
              Column('created_on', DateTime(), default=datetime.now),
              Column('updated_on', DateTime(), default=datetime.now, onupdate=datetime.now)
             )

engine = create_engine('sqlite:///:memory:')

metadata.create_all(engine)
connection = engine.connect()

from sqlalchemy import PrimaryKeyConstraint, UniqueConstraint, CheckConstraint

PrimaryKeyConstraint('user_id', name='user_pk')
UniqueConstraint('username', name='uix_username')
CheckConstraint('unit_cost >= 0.00', name='unit_cost_positive')

from sqlalchemy import select, insert
from sqlalchemy.exc import IntegrityError
ins = insert(users).values(
    username="me",
    email_address="me@me.com",
    phone="111-111-1111",
    password="password_me"
)
try:
    result = connection.execute(ins)
except IntegrityError as error:
    print(error.orig.message, error.params)

s = users.select()
results_1 = connection.execute(s)
for result in results_1:
    print result
    print '-'*50
    print(result.username)
    print '-'*50
    print(result.password)
    

result = connection.execute("select * from users").fetchall()
print result

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    user_id = Column(Integer(), primary_key=True)
    username = Column(String(15), nullable=False, unique=True)
    email_address = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=False)
    password = Column(String(25), nullable=False)
    created_on = Column(DateTime(), default=datetime.now)
    updated_on = Column(DateTime(), default=datetime.now, onupdate=datetime.now)

User.__table__



