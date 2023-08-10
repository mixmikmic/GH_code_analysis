import gzip, csv, sys, os
import __main__ as main

def parse(input_list, dumpfile, output):
    # read each line of dump file and create html for each product in input list
    
    g = gzip.open(dumpfile, 'rb')
    # get list of products to extract
    products = list()
    with open(input_list) as f:
        products = [x.strip() for x in f.readlines()]
    
    # get list of files in output directory
    existing = os.listdir(output)
    
    # iterate dump file
    for l in g:
        line = l.strip()
        # check if line is values(asin, html)
        if line[0] == '(':
            inner = line[1:-2]
            
            # values sperator
            idx = inner.find(',')
            
            # split string into asin and html and convert to strings
            asin = eval(inner[0:idx])
            
            if asin in products and asin + '.html' not in existing:
                html = eval(inner[idx + 1:])
                with open(output + '/{0}.html'.format(asin), mode='w') as f:
                    f.write(html)
                yield (asin)
            

if __name__ == "__main__" and hasattr(main, '__file__'):
    for x in parse(sys.argv[1], sys.argv[2],sys.argv[3]):
        print x
elif __name__ == "__main__":
    for x in parse('asin_Women.csv', 'sampledump.gz',"./test"):
        print x

g = gzip.open('sampledump.gz', 'rb')
[x for x in g][3]

