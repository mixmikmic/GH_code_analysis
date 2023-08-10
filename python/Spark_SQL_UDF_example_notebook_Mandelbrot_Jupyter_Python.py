# the function definition
def mandelbrot(cR, cI, maxIterations):
    zR = cR  
    zI = cI
    i = 1  
    # Iterative formula for Mandelbrot set: z => z^2 + c
    # Escape point: |z|^2 >= 4. Note: z nd c are complex numbers
    while (zR*zR + zI*zI < 4.0 and i < maxIterations):
        newzR = zR*zR - zI*zI +cR
        newzI = 2*zR*zI + cI
        zR = newzR
        zI = newzI
        i += 1
    return i

# registers the function mandelbrot as a UDF for Spark
spark.udf.register("mandelbrot", mandelbrot)

spark.sql("""
with
    x as (select id, -2.0 + 0.027*cast(id as Float) cR from range(0,95)),
    y as (select id, -1.1 + 0.045*cast(id as Float) cI from range(0,50))
select translate(cast(collect_list(substring(' .:::-----++++%%%%@@@@#### ',
       mandelbrot(x.cR, y.cI, 27), 1)) as string), ',', '') as Mandelbrot_Set
from y cross join x 
group by y.id 
order by y.id desc""").show(200, False)

result = spark.sql("""
with
    x as (select id, -2.0 + 0.027*cast(id as Float) cR from range(0,95)),
    y as (select id, -1.1 + 0.045*cast(id as Float) cI from range(0,50))
select translate(cast(collect_list(color) as String), ',' , '') as value
from y cross join x cross join values 
     (0, concat('\u001B','[48;5;0m ','\u001B','[0m')),  -- Black
     (1, concat('\u001B','[48;5;15m ','\u001B','[0m')), -- White
     (2, concat('\u001B','[48;5;51m ','\u001B','[0m')), -- Light blue
     (3, concat('\u001B','[48;5;45m ','\u001B','[0m')), 
     (4, concat('\u001B','[48;5;39m ','\u001B','[0m')),
     (5, concat('\u001B','[48;5;33m ','\u001B','[0m')),
     (6, concat('\u001B','[48;5;27m ','\u001B','[0m')), 
     (7, concat('\u001B','[48;5;21m ','\u001B','[0m'))  -- Dark blue
     as palette(id, color)
where  cast(substring('012223333344445555666677770', mandelbrot(x.cR, y.cI, 27), 1) as Int) = palette.id
group by y.id 
order by y.id desc""").collect()

# print out the result set of the query line by line
for line in result:
    print(line[0])
    



