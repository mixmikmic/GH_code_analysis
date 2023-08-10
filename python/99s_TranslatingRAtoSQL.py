get_ipython().magic('load_ext sql')

get_ipython().magic('sql postgresql://millbr02:@localhost/world')

get_ipython().run_cell_magic('sql', '', '\nselect name, population, surfacearea\nfrom country\nlimit 10')

get_ipython().run_cell_magic('sql', '', '\nselect name as countryname\nfrom country\nlimit 10')

get_ipython().run_cell_magic('sql', '', '\nselect distinct continent\nfrom country')

get_ipython().run_cell_magic('sql', '', '\nselect distinct continent, region\nfrom country\norder by continent')

get_ipython().run_cell_magic('sql', '', '\nselect name, population\nfrom country\nwhere population > 1000000\nlimit 10')

get_ipython().run_cell_magic('sql', '', '\nselect name, population\nfrom country\nwhere population > 1000000 \norder by population desc\nlimit 10')

get_ipython().run_cell_magic('sql', '', "\nselect region, count(*)\nfrom country\nwhere continent = 'Asia'\ngroup by region")

get_ipython().run_cell_magic('sql', '', "\nselect region, sum(surfacearea)\nfrom country\nwhere continent = 'Asia'\ngroup by region\nhaving sum(surfacearea) > 10791100")

get_ipython().run_cell_magic('sql', '', '\nselect city.name, countrycode, country.name, code\nfrom country, city\nlimit 10')

get_ipython().run_cell_magic('sql', '', '\nselect city.name, countrycode, country.name, code\nfrom country, city\nwhere code = countrycode\nlimit 10')

get_ipython().run_cell_magic('sql', '', '\nselect *\nfrom city, country\nwhere countrycode = code\nlimit 10')

