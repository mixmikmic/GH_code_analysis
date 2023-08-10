import time
from datetime import datetime
import sqlite3
import pylab
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

conn = sqlite3.connect("data/content_digest.db") ##Connect to DB
c = conn.cursor() ##Initialize DB Cursor

##Function to Query Database and Return Query as Python List of Lists
def query_db(query):
    start = time.time()
    c.execute(query)
    end = time.time()
    
    print "Query Duration: ",(end-start)/60.0, "Minutes"
    return c.fetchall()

##Funtion to Plot Clicks Over Quarter
def plot_clicks_overall(rows, interval):
    dates = [datetime.strptime(row[0], "%Y-%m-%d") for row in rows]
    rates = [row[1] for row in rows]
    
    plt.plot(dates, rates)
    plt.xticks(dates[::interval], [x.strftime("%m-%d-%Y") for x in dates[::interval]], rotation=45)
    
    plt.tick_params(axis="x", labelsize=20)
    plt.tick_params(axis="y", labelsize=24)

##Function to Plot Clicks Over Quarter by By-Group
def plot_clicks_bygroup(rows, interval):
    for bygroup in list(set(x[1] for x in rows)):
        dates = [datetime.strptime(row[0], "%Y-%m-%d") for row in rows if row[1]==bygroup]
        rates = [row[2] for row in rows if row[1]==bygroup]
        
        plt.plot(dates, rates, label=bygroup)
        plt.xticks(dates[::interval], [x.strftime("%m-%d-%Y") for x in dates[::interval]], rotation=45)

    plt.tick_params(axis="x", labelsize=20)
    plt.tick_params(axis="y", labelsize=24)

##Return Total Number of Links Sent to Users
n = query_db("""SELECT COUNT(*) FROM email_content;""")[0][0]
print "Number of Links: ", n

##Return Total Number of Successful Clicks
n = query_db("""SELECT COUNT(*) FROM clicks WHERE status_code=200;""")[0][0]
print "Number of Clicks: ", n

##Return Total Number of UNIQUE Succesful Clicks (e.g. Page Views)
n = query_db("""SELECT COUNT(*) FROM (SELECT DISTINCT user_id, article_id FROM clicks WHERE status_code=200);""")[0][0]
print "Number of Unique Clicks: ", n

rows_overall = query_db("""
                        SELECT
                            DATE(e.send_time) AS send_date,
                            (100.0*SUM(
                                CASE
                                    WHEN c.user_id NOT NULL THEN 1
                                    ELSE 0
                                END) / COUNT(*)
                            ) AS click_rate
                        FROM 
                            email_content AS e
                                LEFT JOIN (SELECT DISTINCT user_id, article_id FROM clicks WHERE status_code=200) AS c
                                ON e.user_id = c.user_id AND e.article_id = c.article_id
                        GROUP BY send_date
                        ORDER BY send_date
                        ;""")

for row in rows_overall[:5]:
    print row

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plot_clicks_overall(rows_overall, interval=10)
plt.xlabel("Date", fontsize=24)
plt.ylabel("Click Rate (%)", fontsize=24)
plt.title("(A)", fontsize=24)
pylab.ylim([0,50])

plt.subplot(1,2,2)
plot_clicks_overall(rows_overall, interval=10)
plt.xlabel("Date", fontsize=24)
plt.ylabel("Click Rate (%)", fontsize=24)
plt.title("(B)", fontsize=24)
pylab.ylim([15,20])
plt.savefig("graphics/click-rate-overall.png")

rows_by_tod = query_db("""
                        SELECT
                            DATE(e.send_time) AS send_date,
                            (CASE
                                WHEN SUBSTR(TIME(send_time),1,2) IN("08","09","10") THEN "8-11AM"
                                WHEN SUBSTR(TIME(send_time),1,2) IN("11","12","13") THEN "11AM-2PM"
                                WHEN SUBSTR(TIME(send_time),1,2) IN("14","15","16","17") THEN "2PM-6PM"
                            END) AS send_time_of_day, 
                            (100.0*SUM(
                                CASE
                                    WHEN c.user_id NOT NULL THEN 1
                                    ELSE 0
                                END) / COUNT(*)
                            ) AS click_rate
                        FROM 
                            email_content AS e
                                LEFT JOIN (SELECT DISTINCT user_id, article_id FROM clicks WHERE status_code=200) AS c
                                ON e.user_id = c.user_id AND e.article_id = c.article_id
                        GROUP BY send_date, send_time_of_day
                        ORDER BY send_date, send_time_of_day;
                        """)

rows_by_dow = query_db("""
                        SELECT
                            DATE(send_time) as send_date,
                            (CASE 
                                CAST(STRFTIME("%w", send_time) AS INTEGER)
                                    WHEN 0 THEN "Sunday"
                                    WHEN 1 THEN "Monday"
                                    WHEN 2 THEN "Tuesday"
                                    WHEN 3 THEN "Wednesday"
                                    WHEN 4 THEN "Thursday"
                                    WHEN 5 THEN "Friday"
                                    ELSE "Saturday"
                                END) as send_day_of_wk,
                            (100.0*SUM(
                                CASE
                                    WHEN c.user_id NOT NULL THEN 1
                                    ELSE 0
                                END) / COUNT(*)
                            ) AS click_rate
                        FROM 
                            email_content AS e
                                LEFT JOIN (SELECT DISTINCT user_id, article_id FROM clicks WHERE status_code=200) AS c
                                ON e.user_id = c.user_id AND e.article_id = c.article_id
                        GROUP BY send_date, send_day_of_wk
                        ORDER BY send_date, send_day_of_wk
                        ;
                        """)

plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plot_clicks_bygroup(rows_by_dow, interval=2)
plt.xlabel("Date", fontsize=24)
plt.ylabel("Click Rate (%)", fontsize=24)
plt.title("By Day of Week", fontsize=24)
plt.legend(loc="best", fontsize=20)
pylab.ylim([0,50])

plt.subplot(1,2,2)
plot_clicks_bygroup(rows_by_tod, interval=10)
plt.xlabel("Date", fontsize=24)
plt.ylabel("Click Rate (%)", fontsize=24)
plt.title("By Time of Day", fontsize=24)
plt.legend(loc="best", fontsize=20)
pylab.ylim([0,50])

plt.savefig("graphics/click-rate-by-time.png")

rows = query_db("""
                SELECT
                    tp.name AS topic_name,
                    COUNT(*) AS topic_count
                FROM
                    email_content AS e
                        JOIN articles AS a ON e.article_id = a.article_id
                            JOIN topics AS tp ON a.topic_id = tp.topic_id
                GROUP BY topic_name
                ;
                """)

most_common_topics = []
for each in sorted(rows, key=lambda x: x[1], reverse=True)[:8]:
    most_common_topics.append(each[0].encode("ascii"))

print most_common_topics

rows = query_db("""
                SELECT
                    ty.name AS type_name,
                    COUNT(*) AS type_count
                FROM
                    email_content AS e
                        JOIN articles AS a ON e.article_id = a.article_id
                            JOIN types AS ty ON a.type_id = ty.type_id
                GROUP BY type_name
                ;
                """)

most_common_types = []
for each in sorted(rows, key=lambda x: x[1], reverse=True)[:8]:
    most_common_types.append(each[0].encode("ascii"))

print most_common_types

rows_by_topic = query_db("""
                            SELECT
                                DATE(e.send_time) AS send_date,
                                tp.name AS topic_name,
                                (100.0*SUM(
                                    CASE
                                        WHEN c.user_id NOT NULL THEN 1
                                        ELSE 0
                                    END) / COUNT(*)
                                ) AS click_rate
                            FROM
                                email_content AS e
                                    JOIN articles AS a ON e.article_id = a.article_id
                                        JOIN topics AS tp ON a.topic_id = tp.topic_id
                                            LEFT JOIN (SELECT DISTINCT user_id, article_id FROM clicks WHERE status_code=200) AS c
                                            ON e.user_id = c.user_id AND e.article_id = c.article_id
                            GROUP BY send_date, topic_name
                            HAVING COUNT(*) > 500
                            ORDER BY send_date, topic_name
                            ;""")

rows_by_type = query_db("""
                        SELECT
                            DATE(e.send_time) AS send_date,
                            ty.name AS type_name,
                            (100.0*SUM(
                                CASE
                                    WHEN c.user_id NOT NULL THEN 1
                                    ELSE 0
                                END)/ COUNT(*)
                                ) AS click_rate
                            FROM
                                email_content AS e
                                    JOIN articles AS a ON e.article_id = a.article_id
                                        JOIN types AS ty ON a.type_id = ty.type_id
                                            LEFT JOIN (SELECT DISTINCT user_id, article_id FROM clicks WHERE status_code=200) AS c
                                            ON e.user_id = c.user_id AND e.article_id = c.article_id
                        GROUP BY send_date, type_name
                        HAVING COUNT(*) > 500
                        ORDER BY send_date, type_name
                        ;""")

rows_by_common_topic = [x for x in rows_by_topic if x[1] in most_common_topics]
rows_by_common_type =  [x for x in rows_by_type if x[1] in most_common_types ]

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plot_clicks_bygroup(rows_by_common_topic, interval=10)
plt.xlabel("Date", fontsize=24)
plt.ylabel("Click Rate (%)", fontsize=24)
plt.title("By Content Topic (8 Most Common)", fontsize=24)
plt.legend(loc="best", fontsize=18)
pylab.ylim([0,50])

plt.subplot(1,2,2)
plot_clicks_bygroup(rows_by_common_type, interval=10)
plt.xlabel("Date", fontsize=24)
plt.ylabel("Click Rate (%)", fontsize=24)
plt.title("By Content Type (8 Most Common)", fontsize=24)
plt.legend(loc="best", fontsize=18)
pylab.ylim([0,50])

plt.savefig("graphics/clicks-by-content.png")

conn.close() ##Close DB Connection



