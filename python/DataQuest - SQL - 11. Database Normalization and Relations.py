import sqlite3
conn = sqlite3.connect('academy_awards.db')

portman_query = '''select ceremonies.year, nominations.movie from nominations
inner join ceremonies 
on nominations.ceremony_id == ceremonies.id 
where nominations.nominee == "Natalie Portman";
'''
portman_movies = conn.execute(portman_query).fetchall()
for p in portman_movies:
    print(p)

five_movies = conn.execute("select * from movies limit 5;").fetchall()
five_actors = conn.execute("select * from actors limit 5;").fetchall()
five_join_table = conn.execute("select * from movies_actors limit 10;").fetchall()

print(five_movies)
print(five_actors)
print(five_join_table)

q = '''
SELECT actors.actor,movies.movie FROM movies
INNER JOIN movies_actors ON movies.id == movies_actors.movie_id
INNER JOIN actors ON movies_actors.actor_id == actors.id
WHERE movies.movie == "The King's Speech";
'''
kings_actors = conn.execute(q).fetchall()
print(kings_actors)

q = '''
select movies.movie, actors.actor from movies
inner join movies_actors on movies.id == movies_actors.movie_id
inner join actors on actors.id == movies_actors.actor_id
where actors.actor == "Natalie Portman";
'''
portman_joins = conn.execute(q).fetchall()
print(portman_joins)

