from sqlalchemy import *
import datetime

# add `echo=True` to see log statements of all the SQL that is generated
engine = create_engine('sqlite:///:memory:',echo=True) # just save the db in memory for now (ie. not on disk)
metadata = MetaData()
# define a table to use
queries = Table('queries', metadata,
    Column('id', Integer, primary_key=True),
    Column('keywords', String(400), nullable=False),
    Column('timestamp', DateTime, default=datetime.datetime.now),
)
metadata.create_all(engine) # and create the tables in the database

insert_stmt = queries.insert()
str(insert_stmt) # see an example of what this will do

insert_stmt = queries.insert().values(keywords="puppies")
str(insert_stmt)

db_conn = engine.connect()
result = db_conn.execute(insert_stmt)
result.inserted_primary_key # print out the primary key it was assigned

insert_stmt = queries.insert().values(keywords="kittens")
result = db_conn.execute(insert_stmt)
result.inserted_primary_key # print out the primary key it was assigned

from sqlalchemy.sql import select
select_stmt = select([queries])
results = db_conn.execute(select_stmt)
for row in results:
    print row

select_stmt = select([queries]).where(queries.c.id==1)
for row in db_conn.execute(select_stmt):
    print row

select_stmt = select([queries]).where(queries.c.keywords.like('p%'))
for row in db_conn.execute(select_stmt):
    print row

import datetime
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
Base = declarative_base()

class Query(Base):
    __tablename__ = 'queries'
    id = Column(Integer, primary_key=True)
    keywords = Column(String(400))
    timestamp = Column(DateTime,default=datetime.datetime.now)
    def __repr__(self):
        return "<Query(keywords='%s')>" % (self.keywords)
Query.__table__

engine = create_engine('sqlite:///:memory:') # just save the db in memory for now (ie. not on disk)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
my_session = Session()

query = Query(keywords="iguana")
query.keywords

my_session.add(query)
my_session.commit()
query.id

for q in my_session.query(Query).order_by(Query.timestamp):
    print q

query1 = Query(keywords="robot")
query2 = Query(keywords="puppy")
my_session.add_all([query1,query2])
my_session.commit()

for q in my_session.query(Query).order_by(Query.timestamp):
    print q

for q in my_session.query(Query).filter(Query.keywords.like('r%')):
    print q



