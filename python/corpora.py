import codecs
import os

class FileStream(object):
    
    def __init__(self, folder, file_ext=None, encoding='utf-8'):
        self.folder = folder
        self.ext = file_ext
        self.encoding = encoding
        if self.ext is None:
            self.docs = [f for f in os.listdir(self.folder)]
        else:
            self.docs = [f for f in os.listdir(self.folder) if f.endswith(self.ext)]
        
    def __iter__(self):
        for f in self.docs:
            yield self.doc(f)
    
    def doc(self, doc_id):
        with codecs.open(os.sep.join([self.folder, doc_id]), 'rU', encoding=self.encoding) as fs:
            data = fs.read()
        return data 
    
    def first_line(self, doc_id):
        with codecs.open(os.sep.join([self.folder, doc_id]), 'rU', encoding=self.encoding) as fs:
            data = fs.readlines()
        return data[0]

