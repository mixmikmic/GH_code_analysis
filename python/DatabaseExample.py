import os
from sqlalchemy import create_engine
from internal_displacement.model.model import Session

db_url = 'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}'.format(**os.environ)
engine = create_engine(db_url)
Session.configure(bind=engine)
session = Session()

from internal_displacement.model.model import Article, Status

session.query(Article).all()

article = Article(url='http://example.com',
                  domain='example.com',
                  status=Status.NEW)
session.add(article)
session.commit()

session.query(Article).all()

