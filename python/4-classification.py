import metapy

get_ipython().system('wget -N https://raw.githubusercontent.com/meta-toolkit/meta/master/data/lemur-stopwords.txt')

get_ipython().system('wget -N https://meta-toolkit.org/data/2016-01-26/ceeaus.tar.gz')
get_ipython().system('tar xvf ceeaus.tar.gz')

fidx = metapy.index.make_forward_index('ceeaus-config.toml')

fidx.num_labels()

dset = metapy.classify.MulticlassDataset(fidx)
len(dset)

set([dset.label(instance) for instance in dset])

view = dset[0:len(dset)+1]
# or
view = metapy.classify.MulticlassDatasetView(dset)

view.shuffle()
print("{} vs {}".format(view[0].id, dset[0].id))

training = view[0:int(0.75*len(view))]
testing = view[int(0.75*len(view)):len(view)+1]

nb = metapy.classify.NaiveBayes(training)

nb.classify(testing[0].weights)

mtrx = nb.test(testing)
print(mtrx)

mtrx.print_stats()

mtrx = metapy.classify.cross_validate(lambda fold: metapy.classify.NaiveBayes(fold), view, 5)

print(mtrx)
mtrx.print_stats()

ova = metapy.classify.OneVsAll(training, metapy.classify.SGD, loss_id='hinge')

mtrx = ova.test(testing)
print(mtrx)
mtrx.print_stats()

mtrx = metapy.classify.cross_validate(lambda fold: metapy.classify.OneVsAll(fold, metapy.classify.SGD, loss_id='hinge'), view, 5)
print(mtrx)
mtrx.print_stats()

