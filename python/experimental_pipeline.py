import luigi as lg
import json
import pickle

import sys
basedir = '/Users/joewandy/git/lda/code/'
sys.path.append(basedir)

from multifile_feature import SparseFeatureExtractor
from lda import MultiFileVariationalLDA

class ExtractSpectra(lg.Task):

    datadir = lg.Parameter()
    prefix = lg.Parameter()

    def run(self):
        # we could actually extract the spectra from mzxml, mzml files here
        print 'Processing %s and %s' % (datadir, prefix)
    
    def output(self):        
        out_dict = {
            'ms1': lg.LocalTarget(self.datadir + self.prefix + '_ms1.csv'), 
            'ms2': lg.LocalTarget(self.datadir + self.prefix + '_ms2.csv') 
        }
        return out_dict

class GroupFeatures(lg.Task):
    
    scaling_factor = lg.IntParameter(default=1000)
    fragment_grouping_tol = lg.IntParameter(default=7)
    loss_grouping_tol = lg.IntParameter(default=7)
    loss_threshold_min_count = lg.IntParameter(default=5)
    loss_threshold_max_val = lg.IntParameter(default=200)
    loss_threshold_min_val = lg.IntParameter(default=0)

    datadir = lg.Parameter()
    prefixes = lg.ListParameter()
    
    def requires(self):
        return [ExtractSpectra(datadir=datadir, prefix=prefix) for prefix in self.prefixes]
    
    def run(self):

        # input_set is a list of tuples of (ms1, ms2)
        input_set = []
        for out_dict in self.input():
            ms1 = out_dict['ms1'].path
            ms2 = out_dict['ms2'].path
            items = (ms1, ms2)
            input_set.append(items)

        # performs the grouping here
        extractor = SparseFeatureExtractor(input_set, self.fragment_grouping_tol, self.loss_grouping_tol, 
                                           self.loss_threshold_min_count, self.loss_threshold_max_val,
                                           self.loss_threshold_min_val,
                                           input_type='filename')

        fragment_q = extractor.make_fragment_queue()
        fragment_groups = extractor.group_features(fragment_q, extractor.fragment_grouping_tol)

        loss_q = extractor.make_loss_queue()
        loss_groups = extractor.group_features(loss_q, extractor.loss_grouping_tol, 
                                               check_threshold=True)

        extractor.create_counts(fragment_groups, loss_groups, self.scaling_factor)
        mat, vocab, ms1, ms2 = extractor.get_entry(0)
            
        global_word_index = {}
        for i,v in enumerate(vocab):
            global_word_index[v] = i
            
        corpus_dictionary = {}    
        for f in range(extractor.F):
            print "Processing file {}".format(f)
            corpus = {}
            mat, vocab, ms1, ms2 = extractor.get_entry(f)
            n_docs,n_words = mat.shape
            print n_docs,n_words
            d_pos = 0
            for d in ms1.iterrows():
                doc_name = "{}_{}".format(d[1]['mz'],d[1]['rt'])
                corpus[doc_name] = {}
                for word_index,count in zip(mat[d_pos,:].rows[0],mat[d_pos,:].data[0]):
                    if count > 0:
                        corpus[doc_name][vocab[word_index]] = count
                d_pos += 1

            # Added by Simon
            name = input_set[f][0].split('/')[-1].split('ms1')[0][:-1]
            corpus_dictionary[name] = corpus
            
        output_dict = {}
        output_dict['global_word_index'] = global_word_index
        output_dict['corpus_dictionary'] = corpus_dictionary
        with self.output().open('w') as f:
            pickle.dump(output_dict, f)            
            
    def output(self):
        return lg.LocalTarget('output_dict.p')

class RunLDA(lg.Task):

    n_its = lg.IntParameter(default=10)
    K = lg.IntParameter(default=300)
    alpha = lg.FloatParameter(default=1)
    eta = lg.FloatParameter(default=0.1)
    update_alpha = lg.BoolParameter(default=True)
    
    datadir = lg.Parameter()
    prefixes = lg.ListParameter()    

    def requires(self):
        return GroupFeatures(datadir=self.datadir, prefixes=self.prefixes)
    
    def run(self):
        with self.input().open('r') as f:
            output_dict = pickle.load(f)            
        global_word_index = output_dict['global_word_index']
        corpus_dictionary = output_dict['corpus_dictionary']
        mf_lda = MultiFileVariationalLDA(corpus_dictionary, word_index=global_word_index,
                                         K=self.K, alpha=self.alpha, eta=self.eta, 
                                         update_alpha=self.update_alpha)
        mf_lda.run_vb(parallel=False, n_its=self.n_its, initialise=True)

datadir = '/Users/joewandy/Dropbox/Meta_clustering/MS2LDA/large_study/Urine_mzXML_large_study/method_1/POS/'
prefixes = [
    'Urine_StrokeDrugs_02_T10_POS',
    'Urine_StrokeDrugs_03_T10_POS',
    'Urine_StrokeDrugs_08_T10_POS',
    'Urine_StrokeDrugs_09_T10_POS',
]
prefixes_json = json.dumps(prefixes)

lg.run(['RunLDA', '--workers', '1', '--local-scheduler', '--datadir', datadir, '--prefixes', prefixes_json])

