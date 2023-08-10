"""
Takes a filename and returns your data.
For example, with a file that looks like this:
header1, header2, header3
1,2,3
4,5,6

You could get the first row, second header item like this:
dict = get_data("temp.csv")
print dict[1]["header2"]

For the interested, the returned data is a list of dictionaries.  We'll see this more in future weeks.
"""
def get_data(filename):
    filepointer = open(filename, "r")
    data = []
    
    # get_header, inline instead of calling the function above so that the file continues reading
    # from the line right after the header in the for loop below.
    line = filepointer.readline()
    header = line.strip().split(",")

    for line in filepointer:
        fields = line.strip().split(",")

        # Unfortunately, split will split at some commas that we don't mean to split on (e.g., if they've
        # been written into addresses) so we check below to make sure we have the expected number of fields
        # and throw out any other data.  We shouldn't really be throwing out data, we should be fixing the
        # actual problem, but for the purposes of this lab, this will do.
        if (len(fields) == len(header)):
            row = {}
            for fieldNumber in range(len(fields)):
                row[header[fieldNumber]] = fields[fieldNumber]
            data.append(row)
            
    filepointer.close()
    return data

