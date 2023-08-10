widths = ( # These are measurements. Unit is 1 mm.
    13.31,
    15.56,
    7.05,
    12.97,
    15.18,
    16.22,
    17.74,
    14.85,
    0,
    15.08,
    6.99,
    10.13,
    8.26,
    0,
    11.41,
    9.76,
)

for x in widths:
    sae = x / 25.4
    print(x, sae*2, sae*4, sae*8, sae*16)

