import pstats
p = pstats.Stats('output_serial.prof')
p.print_stats()

p.strip_dirs()
p.sort_stats('ncalls')
p.print_stats(20)

p.strip_dirs()
p.sort_stats('cumtime')
p.print_stats(10)

p.strip_dirs()
p.sort_stats('tottime')
p.print_stats(10)

p.print_stats("lennard_jones_potential.py:")

p1 = pstats.Stats('output_parallel_1.prof')
p1.strip_dirs()
p1.sort_stats('tottime')
p1.print_stats('LJ_multi_1.py|lennard_jones_potential.py')

p1.sort_stats('cumtime')
p1.print_stats('LJ_multi_1.py|lennard_jones_potential.py')

