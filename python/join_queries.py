import sys
import sqlalchemy as sa
import sqlalchemy.orm as sa_orm
import testing.postgresql

from app import models
from app.util import sqldebug

# open a test session
postgresql = testing.postgresql.Postgresql(base_dir='.test_db')
db_engine = sa.create_engine(postgresql.url())
models.init_database(db_engine)
sessionmaker = sa_orm.sessionmaker(db_engine)
session = sessionmaker()

simple_query = session.query(models.Submission)
sqldebug.pp_query(simple_query)

current_joined_query = session.query(models.Submission)    .options(
        sa_orm.joinedload(models.Submission.user, innerjoin=True)
    )
sqldebug.pp_query(current_joined_query)

new_joined_query = session.query(models.Submission)    .join(models.User)    .join(models.Form)    .options(
        sa_orm.contains_eager(models.Submission.user),
        sa_orm.contains_eager(models.Submission.form).load_only('name'),
    )
    
sqldebug.pp_query(new_joined_query)



