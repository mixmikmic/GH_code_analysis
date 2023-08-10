with open('movie_metadata.csv') as file:
    data = [ line.strip().split(',') for line in file.readlines()[1:] ]

len(data)

data[0]

movie_title_and_year = map(lambda x: (x[11].decode("ascii" , "ignore"), x[23]), data)

len(movie_title_and_year)

movie_title_and_year[:10]

james_cameron_movies = filter(lambda x: x[1] == 'James Cameron', data)

len(james_cameron_movies)

james_cameron_movies[0]

count = reduce(lambda x, y: x+1, data, 0)

count

map(lambda x: x[11].decode("ascii" , "ignore"), filter(lambda x: x[1] == 'James Cameron', data))

reduce(lambda x, y: x+y, map(lambda x: len(x[9].split('|')), data))

sum_num = reduce(
    lambda x, y: (x[0]+y, x[1]+1),
    map(
        lambda x: int(x[22]) if x[22].isdigit() else 0,
        filter(
            lambda x: x[1] == 'James Cameron',
            data)
    ),
    (0, 0)
)

sum_num

round(float(sum_num[0]) / sum_num[1], 2)

import csv

with open('movie_metadata.csv') as file:
    sum_num = reduce(
        lambda x, y: (x[0]+y, x[1]+1),
        map(
            lambda x: int(x[22]) if x[22].isdigit() else 0,
            filter(
                lambda x: x[1] == 'James Cameron',
                csv.reader(iter(file.readline, ''))
            )
        ),
        (0, 0)
    )

sum_num

