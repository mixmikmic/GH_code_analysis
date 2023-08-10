from BaseXClient import BaseXClient

session = BaseXClient.Session('127.0.0.1', 1984, 'admin', 'admin')

publisher = '//rec/df[@t="210"]/sf[@c="c"]'
publisher_city = '//rec/df[@t="210"]/sf[@c="a"]'
subject = '//rec/df[@t="606"]/sf[@c="a"]'

def count(path, limit):
    q = '''let $db := db:open("bni")
    let $result :=
        for $publisher in distinct-values($db{0})
              let $count :=  count(index-of($db{0}, $publisher))
              order by $count descending
        return concat($publisher, ", ", $count)
    for $limited at $lim in subsequence($result, 1, {1})          
        return $limited'''.format(path, limit)
    query = session.query(q)
    for _, item in query.iter():
        print(item)

count(publisher, 10)

count(publisher_city, 10)

count(subject, 30)

