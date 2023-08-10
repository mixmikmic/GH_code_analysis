

# Elastic search library
from elasticsearch import Elasticsearch, helpers
# python-mysql database access
import MySQLdb
import logging
import pandas
from ConfigParser import ConfigParser

config = ConfigParser()
config.read("settings")

args = {}
# There are two sections: mysql and elasticsearch
if config.has_section("mysql"):
    if config.has_option("mysql", "user") and         config.has_option("mysql", "password") and         config.has_option("mysql", "mlstats_db") and         config.has_option("mysql", "cvsanaly_db") and         config.has_option("mysql", "code_review_db"):
        args["mysql"] = dict(config.items("mysql"))

if config.has_section("elasticsearch"):
    if config.has_option("elasticsearch", "user") and        config.has_option("elasticsearch", "password") and        config.has_option("elasticsearch", "host") and        config.has_option("elasticsearch", "port") and        config.has_option("elasticsearch", "path"):
         args["elasticsearch"] = dict(config.items("elasticsearch"))

if not(args.has_key("mysql") and args.has_key("elasticsearch")):
    raise Exception("Section 'mysql' or section 'elasticsearch' not found in the 'settings' file")





def connect(args):
   user = args["mysql"]["user"]
   password = args["mysql"]["user"]
   host = "localhost"
   db = args["mysql"]["code_review_db"]

   try:
      db = MySQLdb.connect(user = user, passwd = password, db = db, charset='utf8')
      return db, db.cursor()
   except:
      logging.error("Database connection error")
      raise
        


def execute_query(connector, query):
   results = int (connector.execute(query))
   cont = 0
   if results > 0:
      result1 = connector.fetchall()
      return result1
   else:
      return []
    

db, cursor = connect(args)

# Insert data in ElasticSearch
def to_json(row, columns):
    # Function that translates from tuple to JSON doc
    doc = {}
    
    for column in columns:
       
        value = row[columns.index(column) + 1] 
       
        try:
            doc[column] = value
        except:
            doc[column] = ""
   
    return doc

query_patchseries = """ SELECT ps.id as patchserie_id,
                                -1 as patch_id,
                                -1 as comment_id,
                                ps.subject as subject,
                                ps.message_id as message_id,
                                pe.email as sender,
                                SUBSTRING_INDEX(pe.email, '@', -1) as sender_domain,
                                MIN(psv.date_utc) as sent_date,
                                0 as balance,
                                'na' as flag,
                                IF(p.commit_id IS NULL, 0, 1) as merged,
                                'patchserie' as emailtype,
                                0 as num_flag_review,
                                0 as num_flag_ack,
                                0 as num_patches,
                                -1 as is_acked,
                                -1 as post_ack_comment,
                                0 as patchserie_numpatches,
                                0 as patchserie_numackedpatches,
                                0 as patchserie_percentage_ackedpatches
                         FROM  patch_series ps,
                               patch_series_version psv,
                               patches p,
                               people pe
                         WHERE pe.id = p.submitter_id AND
                               p.ps_version_id = psv.id AND
                               psv.ps_id = ps.id
                         GROUP BY ps.id """
                               
                         
query_patches = """ SELECT psv.ps_id as patchserie_id,
                           p.id as patch_id,
                           -1 as comment_id,
                           p.subject as subject,
                           p.message_id as message_id,
                           pe.email as sender,
                           SUBSTRING_INDEX(pe.email, '@', -1) as sender_domain,
                           p.date_utc as sent_date,
                           -1 as balance,
                           'na' as flag,
                           IF(p.commit_id IS NULL, 0, 1) as merged,
                           'patch' as emailtype,
                           0 as num_flag_review,
                           0 as num_flag_ack,
                           1 as num_patch,
                           IF(t.flag='Acked-by', 1, 0) as is_acked,
                           0 as post_ack_comment,
                           -1 as patchserie_numpatches,
                           -1 as patchserie_numackedpatches,
                           -1 as patchserie_percentage_ackedpatches
                    FROM   patch_series_version psv,
                           patches p,
                           people pe,
                           (SELECT p.id as patch_id,
                                   f.flag as flag
                            FROM patches p
                            LEFT JOIN flags f
                            ON p.id = f.patch_id AND 
                               f.flag = 'Acked-by') t
                    WHERE p.submitter_id = pe.id AND
                          psv.id = p.ps_version_id AND
                          p.id = t.patch_id"""
                    
                           

query_flags = """ SELECT psv.ps_id as patchserie_id,
                         patch_id as patch_id,
                         -1 as comment_id,
                         p.subject as subject,
                         'na' as message_id,
                         SUBSTRING_INDEX(SUBSTRING_INDEX(value, '<', -1), '>', 1) as sender,
                         SUBSTRING_INDEX(SUBSTRING_INDEX(SUBSTRING_INDEX(value, '<', -1), '>', 1), '@', -1) as sender_domain,
                         f.date_utc as sent_date,
                         IF(flag = 'Reviewed-by', 1, 0) as balance,
                         flag as flag,
                         -1 as merged,
                         'flag' as emailtype,
                         IF(flag = 'Reviewed-by', 1, 0) as num_flag_review,
                         IF(flag = 'Acked-by', 1, 0) as num_flag_ack,
                         0 as num_patch,
                         -1 as is_acked,
                         -1 as post_ack_comment,
                         -1 as patchserie_numpatches,
                         -1 as patchserie_numackedpatches,
                         -1 as patchserie_percentage_ackedpatches
                  FROM patch_series_version psv,
                       patches p,
                       flags f
                  WHERE psv.id = p.ps_version_id AND
                        p.id = f.patch_id
                 """

#If required, some example of filters by flag
#(flag = 'Reviewed-by' OR
# flag = 'Acked-by')


query_comments = """ SELECT psv.ps_id as patchserie_id,
                            c.patch_id as patch_id,
                            c.id as comment_id,
                            c.subject as subject,
                            c.message_id as message_id,
                            pe.email as sender,
                            SUBSTRING_INDEX(pe.email, '@', -1) as sender_domain,
                            c.date_utc as sent_date,
                            0 as balance,
                            'na' as flag,
                            -1 as merged,
                            'comment' as emailtype,
                            0 as num_flag_review,
                            0 as num_flag_ack,
                            0 as num_patch,
                            -1 as is_acked,
                            0 as post_ack_comment,
                            -1 as patchserie_numpatches,
                            -1 as patchserie_numackedpatches,
                            -1 as patchserie_percentage_ackedpatches
                     FROM patch_series_version psv,
                          patches p,
                          people pe,
                          comments c
                     WHERE psv.id = p.ps_version_id AND
                           p.id = c.patch_id AND
                           c.submitter_id = pe.id """
                            

# How to test post-ack comments:
# select comments.id, subject, date, first_ack_date, comments.patch_id 
# from comments, (select patch_id, min(date) as first_ack_date from flags where flag='Acked-by' group by patch_id) t 
# where date > t.first_ack_date and t.patch_id = comments.patch_id limit 100;

query_post_ack = """ SELECT comments.id as comment_id,
                            1 as post_ack_comment
                     FROM comments left join 
                              (SELECT patch_id,
                                      MIN(date) as first_ack_date 
                               FROM flags 
                               WHERE flag='Acked-by' 
                               GROUP BY patch_id) t 
                           ON t.patch_id=comments.patch_id 
                     WHERE date > t.first_ack_date """

# Query to detect those comments that are sent by the same developer that sent the original patch
query_self_comment = """ SELECT c.id as comment_id,
                                 'self-comment' as emailtype
                          FROM comments c,
                               patches p
                          WHERE c.submitter_id = p.submitter_id AND
                                p.id = c.patch_id
                          ORDER BY comment_id """

# Query to detect the number of ack-ed patches from a patchserie
query_acked_patches = """ SELECT psv.ps_id as patchserie_id,
                                 IF(count(distinct(p.series)) = 0, 1, count(distinct(p.series))) as acked_patches,
                                 t1.total_num_patches as total_num_patches,
                                 IFNULL(TRUNCATE(((IF(count(distinct(p.series)) = 0, 1, count(distinct(p.series)))/t1.total_num_patches)*100), 0), 0) as percentage_acked_patches
                          FROM patch_series_version psv,
                               patches p,
                               flags f,
                               (SELECT psv.ps_id,
                                       IF(count(distinct(p.series))=0, 1, count(distinct(p.series)))  as total_num_patches
                                FROM patch_series_version psv,
                                     patches p
                                WHERE psv.id=p.ps_version_id 
                                GROUP BY psv.ps_id) t1 
                           WHERE f.patch_id=p.id AND  
                                 f.flag='Acked-by' AND
                                 psv.id=p.ps_version_id AND
                                 psv.ps_id=t1.ps_id 
                           GROUP BY psv.ps_id """

data_patchseries = list(execute_query(cursor, query_patchseries))
data_patches = list(execute_query(cursor, query_patches))
data_flags = list(execute_query(cursor, query_flags))
data_comments = list(execute_query(cursor, query_comments))
data_post_ack = list(execute_query(cursor, query_post_ack))
data_self_comment = list(execute_query(cursor, query_self_comment))
data_acked_patches = list(execute_query(cursor, query_acked_patches))

columns = ["patchserie_id", "patch_id", "comment_id", "subject", "message_id", "sender", "sender_domain", "sent_date", "balance", "flag", "merged", "emailtype", "num_flag_review", "num_flag_ack", "num_patch", "is_acked", "post_ack_comment", "patchserie_numpatches", "patchserie_numackedpatches", "patchserie_percentage_ackedpatches"]

patchseries_df = pandas.DataFrame(data_patchseries, columns = columns)
patches_df = pandas.DataFrame(data_patches, columns = columns)
flags_df = pandas.DataFrame(data_flags, columns = columns)
comments_df = pandas.DataFrame(data_comments, columns=columns)
post_ack_df = pandas.DataFrame(data_post_ack, columns=["comment_id", "post_ack_comment"])
self_comments_df = pandas.DataFrame(data_self_comment, columns=["comment_id", "emailtype"])
acked_patches_df = pandas.DataFrame(data_acked_patches, columns=["patchserie_id",  "patchserie_numackedpatches", "patchserie_numpatches", "patchserie_percentage_ackedpatches"])

# 1. Update values for the post_ack_comments

# Set index to comment_id
res_comments = comments_df.set_index("comment_id")
res_post_ack = post_ack_df.set_index("comment_id")
res_self_comments = self_comments_df.set_index("comment_id")

# Let's update the comments index with the post_ack comments
res_comments.update(res_post_ack)
# Let's update the comments index with the self_comments data
res_comments.update(res_self_comments)

# Reset index and order columns as expected prior the concat action
reseted_index_comments = res_comments.reset_index()
# Check this properly worked
# reseted_index_comments[reseted_index_comments["post_ack_comment"]==1]

# Then I need to move two columns the comments_id column.
reseted_index_comments = reseted_index_comments[columns]
reseted_index_comments

# 3. Update values for the acked_patches analysis
res_patchseries = patchseries_df.set_index("patchserie_id")
res_acked_patches = acked_patches_df.set_index("patchserie_id")

# mix  both dataframes
res_patchseries.update(res_acked_patches)

# reset index to a generic one
reseted_index_patchseries = res_patchseries.reset_index()
# order again the columns
reseted_index_patchseries = reseted_index_patchseries[columns]

all_data = pandas.concat([reseted_index_patchseries, patches_df, flags_df, reseted_index_comments])

INDEX = 'xen-patchseries-reviewers'

# Adding specific not-analyzed requirements to a couple of fields
mapping = '''
{
  
       "properties":{
           "sender": {
               "type":     "string",
               "index":    "not_analyzed"
           },
           "sender_domain": {
               "type":     "string",
               "index":    "not_analyzed"
           }
           
       }
    
}
'''

#As the mapping does not seem to work, we need to PUT the following:
"""
PUT /xen-patchseries-reviewers/_mapping/patchserie
{
    "properties": {
        "sender" : {
            "type": "string",
            "index": "not_analyzed"
        }
    }
}
"""

# Building the ES connection
user = args["elasticsearch"]["user"]
password = args["elasticsearch"]["password"]
host = args["elasticsearch"]["host"]
port = args["elasticsearch"]["port"]
path = args["elasticsearch"]["path"]
connection = "http://" + user + ":" + password + "@" + host + ":" + port + "/" + path

es2 = Elasticsearch([connection])
# Creating the index
#es2.indices.create(index=INDEX, body=mapping)

columns = all_data.columns.values.tolist()
print columns
uniq_id = 0
bulk_doc = []
for row in all_data.itertuples():
    # Let's insert into ES each tuple found in the dataset
    uniq_id = uniq_id + 1
    doc = to_json(row, columns)
    header = {
        "_index": INDEX,
        "_type": "patchserie",
        "_id": uniq_id,
        "_source": doc
    }
    
    bulk_doc.append(header)
    if uniq_id % 5000 == 0:
        helpers.bulk(es2, bulk_doc)
        bulk_doc = []
    


helpers.bulk(es2, bulk_doc)



