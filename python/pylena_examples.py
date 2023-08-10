import pylena

lena_file = pylena.LenaFile("93_07_lena5min.csv")

print "child_key:   " + lena_file.child_key
print "birth_date:  " + lena_file.birth_date
print "age:         " + lena_file.age 
print "sex:         " + lena_file.sex

rows_1_to_7 = lena_file.get_range(begin=1, end=7) # from rows 1 to 7 (does not include row 7)

print rows_1_to_7.sum("ctc")

print rows_1_to_7.sum("ctc", "cvc")

print rows_1_to_7.total_time()

print rows_1_to_7.total_time(begin=2, end=5) # from rows 2 to 5 (does not include row 5)

print lena_file.total_time()

print lena_file.total_time(begin=0, end=22) # from rows 0 to 22 (does not include row 22)

print rows_1_to_7.range

for element in rows_1_to_7.range:
    print element.timestamp

for element in rows_1_to_7.range:
    print element.awc_actual

row_1 = rows_1_to_7.range[0]

import pprint
pp = pprint.PrettyPrinter(indent=3)
# ^ ignore this, it's just for printing the dictionary nicely.

pp.pprint(row_1.__dict__)

print "duration: " + row_1.duration
print "awc_actual: " + str(row_1.awc_actual) # have to cast int to string here
print "birth_date: " + row_1.birth_date
print "processing_file: " + row_1.processing_file

ranked_ctc = lena_file.rank_window(6, "ctc")

print ranked_ctc[:5]

ranked_ctc_cvc = lena_file.rank_window(6, "ctc", "cvc")

print ranked_ctc_cvc[:5]

top_6_rows = lena_file.top_rows(6, "cvc")

print top_6_rows

# this is a list comprehension. It basically says take the first 
# element of each tuple in the list and make a new list with them.
the_indices = [element[0] for element in top_6_rows] 

print "the_indices: "
print the_indices


top_6_LenaRows = lena_file.get_rows(rows=the_indices)


print "\nthe top_6_LenaRows: "

pp.pprint(top_6_LenaRows)

 
print "\nthe cvc values: \n"

for row in top_6_LenaRows:
    print row.cvc_actual

