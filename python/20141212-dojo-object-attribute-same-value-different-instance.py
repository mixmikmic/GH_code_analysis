class FordFiesta():
    pass

    def __init__(self, color='black'):
        self.color = color

my_car = FordFiesta()
my_car, my_car.color

your_car = FordFiesta()
your_car, your_car.color

my_car is your_car

my_car.color is your_car.color

another_car = FordFiesta('black')
another_car, another_car.color

my_car.color is another_car.color

s = '''
;lksa;lkjarpoeit;lkfdn;sh wv;omriwjglirns;dfkjewwww;slkmjes;rjtr.lksd;ls
;lsakjda ;rlkjwre;gijwpojls hwnsdfkjdf vkihgkesaf sfdknkjfiugrlkjfdlkjh 
dlkdslklkdsldspoirew98u5c3498mu98mu4y6vphegwijgew oijgtwijgtrpoijwnfds
mpu wgec3pe3pitcwgpi,iiiiiy5epoijgtroijreoijrewpoijtrpojtrtrpojjpjotrpo
 wwj gjgtrcjtcppjgtcpojiiire;oijse;roij;oreijger;oijer;glkjrgf;lkjdg;sd
 pwqeiupqoewiuwpetiuwepoiuwerpoupwiurwpeoriuewporiuewrpoiuewrpowutpweiu
 p98u34p08u409324u70439840987324029387520394872043987324098237409238745
 '''

s[:]

my_car = FordFiesta(s[:])
my_car, my_car.color

your_car = FordFiesta(s[:])
your_car, your_car.color

my_car is your_car

my_car.color is your_car.color

my_car = FordFiesta(s[:] + 'just one bite more')
my_car, my_car.color

your_car = FordFiesta(s[:] + 'just one bite more')
your_car, your_car.color

my_car is your_car

my_car.color == your_car.color

my_car.color is your_car.color

id(my_car.color), id(your_car.color)

