get_ipython().run_cell_magic('writefile', 'tmp/PlantExpansion.mod', '\nset PLANTS;                            # Set of plant types\nset DEMAND;                            # Demand Segments\nset SCENARIOS;                         # Planning Scenarios\n\nparam e{PLANTS};                       # Current Plant Capacity\nparam C{PLANTS};                       # Capital Cost per unit Expansion\nparam O{PLANTS};                       # Operating Cost [k$/GWh]\n\nparam T{DEMAND};                       # Time Periods for Demand Segments\nparam D{DEMAND,SCENARIOS};             # Demand Scenarios\n\nvar x {PLANTS} >= 0;                   # Plant Expansion\nvar y {PLANTS,DEMAND,SCENARIOS} >= 0;  # Operating Schedule\nvar v {SCENARIOS};                     # Variable Cost\nvar capcost;                           # Capital Cost\n\nminimize COST: capcost + sum {s in SCENARIOS} 0.25*v[s];\n\ns.t. CAPCOST: capcost = sum{p in PLANTS} C[p]*(e[p]+x[p]);\ns.t. VARCOST {s in SCENARIOS}:\n   v[s] = sum {p in PLANTS, d in DEMAND} T[d]*O[p]*y[p,d,s];\ns.t. DEMANDS {p in PLANTS, s in SCENARIOS}: \n   e[p] + x[p] >= sum {d in DEMAND} y[p,d,s];\ns.t. C4 {d in DEMAND, s in SCENARIOS} :\n   D[d,s] = sum {p in PLANTS} y[p,d,s];\n   \nsolve;\n\ntable results {p in PLANTS} OUT "CSV" "tmp/PlantExpansion.csv" "Table" :\n    p~Plant,\n    O[p]~Unit_Cost,\n    e[p]~Current_Cap,\n    x[p]~Exp_Cap,\n    x[p]+e[p]~Total_Cap;\n\nend;')

get_ipython().run_cell_magic('script', 'glpsol -m tmp/PlantExpansion.mod -d /dev/stdin -y tmp/PlantExpansion.txt --out output', '\nset SCENARIOS := S1 S2 S3 S4;\n\nparam: DEMAND: T :=\n    Base      24\n    Peak       6 ;\n\nparam: PLANTS:     e     C     O:=\n    Coal        1.75   200    30\n    Hydro       2.00   500    10\n    Nuclear     0.00   300    20\n    Grid        0.00     0   200 ;\n\nparam D :   S1     S2    S3    S4 :=\n    Base   8.25   10.0  7.50  9.00\n    Peak   2.50   2.00  2.50  1.50 ;\n\nend;')

print output

import pandas

results = pandas.read_csv("tmp/PlantExpansion.csv")
results





