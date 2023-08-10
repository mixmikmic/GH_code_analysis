get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')

import os
import json
import re
from itertools import chain
from semproc.yaml_configs import import_yaml_configs
from semproc.parser import Parser
from semproc.rawresponse import RawResponse

class Identify():
    '''
    parameters:
        yaml_file: path to the yaml definition yaml
        source_content: the content string for comparisons
        source_url: the url string for comparisons
        options: dict containing the filtering options, ie
                 identify which protocol, identify which service
                 of a protocol, identify if it's a dataset service
                 for a protocol
    '''
    def __init__(self, yaml_files, source_content, source_url):
        '''
        **options:
            parser: Parser from source_content
            ignore_case: bool
        '''
        self.source_content = source_content
        self.source_url = source_url
        self.yaml = import_yaml_configs(yaml_files)
        
        self.parser = Parser(source_content)
        

    def _filter(self, operator, filters, clauses):
        '''
        generate a list of dicts for operator and booleans
        that can be rolled up into some bool for a match
        '''
        for f in filters:
            filter_type = f['type']

            if filter_type == 'complex':
                filter_operator = f['operator']
                clauses.append(self._filter(filter_operator, f['filters'], []))
            elif filter_type == 'simple':
                filter_object = self.source_content if f['object'] == 'content' else self.source_url
                filter_value = f['value']

                # TODO: a better solution than this
                filter_value = filter_value.upper()
                filter_object = filter_object.upper()
                
                clauses.append(filter_value in filter_object)
            elif filter_type == 'regex':
                filter_object = self.source_content if f['object'] == 'content' else self.source_url
                filter_value = f['value']
                clauses.append(len(re.findall(filter_value, filter_object)) > 0)
            elif filter_type == 'xpath':
                # if the filter is xpath, we can only run against
                # the provided xml (parser) and ONLY evaluate for existence
                # ie the xpath returned some element, list, text value
                # but we don't care what it returned
                xpath = f['value']
                if not self.parser:
                    # nothing to find, this is an incorrect filter
                    clauses.append(False)

                # try the xpath but there could be namespace or
                # other issues (also false negatives!)
                try:
                    clause = self.parser.xml.xpath(xpath) not in [None, '', []]
                except:
                    clause = False

                clauses.append(clause)

        return {operator: clauses}

    def _evaluate(self, clauses, sums):
        '''
        evaluate a list a dicts where the key is
        the operator and the value is a list of
        booleans
        '''
        if isinstance(clauses, bool):
            # so this should be the rolled up value
            return clauses

        for k, v in clauses.iteritems():
            if isinstance(v, dict):
                return sums + self._evaluate(v, 0)
            elif isinstance(v, list) and not all(isinstance(i, bool) for i in v):
                # TODO: this is not a good assumption
                for i in v:
                    sums += self._evaluate(i, 0)
                return sums
            if k == 'ands':
                # everything must be true
                sums += sum(v) == len(v)
            elif k == 'ors':
                # any one must be true
                sums += sum(v) > 0

        return sums

    def identify(self):
        '''
        it is within a protocol if *any* set of filters
        '''
        def _test_option(filters):
            '''where filters is the set of filters as booleans'''
            if not filters:
                return False
            
            for i, j in filters.iteritems():
                if self._evaluate({i: self._filter(i, j, [])}, 0):
                    return True
                
            return False
        
        def _extract_option(filters):
            '''
            where filters is the set of things to return a value
            this assumes that you have concatenated the defaults and/or checks set
            '''
            if not filters:
                return []
            
            items = []
            for check in filters:
                for c in check[1]:
                    item = ''
                    if c['type'] == 'simple':
                        # TODO: this is still not a safe assumption re: casing
                        filter_value = c['value'].upper()
                        filter_object = self.source_content if c['object'] == 'content' else self.source_url
                        filter_object = filter_object.upper()
                        
                        if filter_value in filter_object:
                            item = [c.get('text', '')]  # just for the xpath handling later
                    elif c['type'] == 'xpath':
                        if not self.parser.xml:
                            print 'Parser FAIL'
                            continue
                        
                        try:
                            values = self.parser.xml.xpath(c['value'])
                            item = [v.strip() for v in values if v is not None]
                        except Exception as ex:
                            print 'XPATH FAIL: ', ex 
                            print c['value']
                            continue
                    
                    if item:
                        items += item
            
            return items
        
        def _chain(source_dict, keys):
            if not source_dict:
                return []
            return list(chain.from_iterable(
                    [source_dict.get(key, {}).items() for key in keys]
                ))
        
        matches = []
        for protocol in self.yaml:
            protocol_name = protocol['name']
            # print protocol_name

            for k, v in protocol.iteritems():
                if k in ['name'] or v is None:
                    continue

                for option in v:
                    is_match = _test_option(option['filters'])
                            
                    # check the error filters
                    errors = option.get('errors', {})
                    is_error = _test_option(errors.get('filters', {})) if errors else False

                    # check the language filters
                    language_filters = option.get('language', {})
                    _filters = _chain(language_filters, ["defaults", "checks"])
                    languages = _extract_option(_filters)

                    # check the version filters
                    version_filters = option.get('versions', {})
                    _filters = _chain(version_filters, ["defaults", "checks"])
                    versions = _extract_option(_filters)
                    
                    # and the dialect if there's a key
                    dialect_filters = option.get('dialect', {})
                    if dialect_filters:
                        if 'text' in dialect_filters:
                            dialect = dialect_filters.get('text')
                        else:
                            # it's in the response somewhere
                            _filters = _chain(dialect_filters, ["defaults", "checks"])
                            dialect = _extract_option(_filters)
                    

                    # dump it out
                    if is_match:
                        matches.append({
                                "protocol": protocol_name, 
                                k: {
                                    "name": option.get('name', ''),
                                    "request": option.get('request', ''),
                                    "dialect": dialect,
                                    "version": versions,
                                    "error": is_error,
                                    "language": languages
                                }
                            })

        return matches

import glob

def _prep(filepath):
    with open(filepath, 'r') as f:
        response = f.read()

    response = response.replace('\\\n', '').replace('\r\n', '').replace('\\r', '').replace('\\n', '').replace('\n', '')
    return response.decode('utf-8', errors='replace').encode('unicode_escape')

# print response

#yamls = glob.glob('../semproc/configs/*_identifier.yaml')

yamls = yamls = ['../semproc/configs/oaipmh_identifier.yaml']
url = 'http://www.example.com?verb=ListRecords'
response = _prep('../response_examples/oaipmh_listrecords.xml')

# url = 'http://www.example.com/opensearch.xml'
# response = _prep('../response_examples/opensearch_blended_parameters.xml')
# response = _prep('../response_examples/opensearch_usgs_search_atom.xml')

# yamls = ['../semproc/configs/opensearch_identifier.yaml', 
#          '../semproc/configs/iso_identifier.yaml']

identifier = Identify(yamls, response, url)
identifier.identify()

from lxml import etree

text = '''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:georss="http://www.georss.org/georss"
 xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
 <title>ScienceBase search results</title>
 <author>
  <name>USGS ScienceBase</name>
  <uri>http://www.sciencebase.gov/</uri>
 </author></feed>
'''

# this is ridiculous but it functions and won't be used (*shrugs*)
def check_count(context, test, if_true=True, if_false=False):
    print test, type(test)
    return if_true if test else if_false
ns = etree.FunctionNamespace(None)
ns['check_count'] = check_count


xml = etree.fromstring(text)

# xml.xpath('check_count(count(/*/namespace::*[. = "http://a9.com/-/spec/opensearch/1.1/"]) = count(/*/namespace::*))')

x = xml.xpath('count(/*/namespace::*[. = "http://a9.com/-/spec/opensearch/1.1/"]) < 1')

'''
- type: xpath-function
    object: content
    # make this some xpath expression that evaluates 
    # with a function *in* the code (not this one though it is not useful at all)
    value: 'check_count(count(/*/namespace::*[. = "http://a9.com/-/spec/opensearch/1.1/"]) = count(/*/namespace::*)'
    if_true: True
    if_false: False 
'''

print x





