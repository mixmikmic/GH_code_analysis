# Create a number in scientific notation
value_scientific_notation =  6.32000000e-03

# Create a vector of number in scientific notation
vector_scientific_notation = [6.32000000e-03,
                              1.80000000e+01,
                              2.31000000e+00,
                              0.00000000e+00,
                              5.38000000e-01,
                              6.57500000e+00,
                              6.52000000e+01,
                              4.09000000e+00,
                              1.00000000e+00,
                              2.96000000e+02,
                              1.53000000e+01,
                              3.96900000e+02,
                              4.98000000e+00]

# Display values as float
'{:f}'.format(value_scientific_notation)

# Display vector values as floats
['{:f}'.format(x) for x in vector_scientific_notation]

