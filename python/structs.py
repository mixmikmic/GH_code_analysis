import struct

data = struct.pack("id5s", 1, 2.2, "THREE")
print data
print map(ord, data)

struct.unpack("id5s", data)

import numpy
array = numpy.empty(100, dtype=[("one", numpy.int32),
                                ("two", numpy.float64),
                                ("three", "|S5")])

array["one"][0] = 1
array["two"][0] = 2.2
array["three"][0] = "THREE"

array[0]

array.view("S1")[: 4 + 8 + 5]

array = numpy.empty(100, dtype=numpy.dtype(
    [("one", numpy.int32),
     ("two", numpy.float64),
     ("three", "|S5")], align=True))

array["one"][0] = 1
array["two"][0] = 2.2
array["three"][0] = "THREE"

array.view("S1")[:len(data)]

