from makammusicbrainz.audiometadata import AudioMetadata
from ahenkidentifier.ahenkidentifier import AhenkIdentifier
import json
import os

anno_dict = json.load(open('../annotations.json'))
mbids = anno_dict.keys()
audio_meta_crawler = AudioMetadata()

for ii, mbid in enumerate(mbids):
    save_file = os.path.join('../metadata/%s.json' % mbid)
    if not os.path.exists(save_file):
        print("Crawling %s" %mbid)
        meta = audio_meta_crawler.from_musicbrainz(mbid)
        json.dump(meta, open(save_file, 'w'), indent=4)
        

tonic_dict = AhenkIdentifier._get_dict('tonic')  # load the makam tonic dict

for ii, (key, val) in enumerate(anno_dict.items()):
    # load the metadata
    meta_file = os.path.join('../metadata/%s.json' % key)
    meta = json.load(open(meta_file))
    
    try:
        makams = set(mm['attribute_key'] for mm in meta['makam'])

        tonics = set(AhenkIdentifier._get_tonic_symbol_from_makam(
                mm, tonic_dict)[0] for mm in makams)

        if len(tonics) == 1:
            for aa in anno_dict[key]['annotations']:
                if aa['tonic_symbol']:  # don't override
                    if aa['tonic_symbol'] != list(tonics)[0]:
                        print str(ii) + ' ' + key
                        print "... Tonic symbol mismatch: " + aa['tonic_symbol'] + ' -> ' + list(tonics)[0]
                else:
                    aa['tonic_symbol'] = list(tonics)[0]
        elif len(tonics) > 1:
            print str(ii) + ' ' + key
            print "... More than one tonic symbol for: " + ', '.join(makams)
        else:  # not tonic
            print str(ii) + ' ' + key
            print "... Tonic symbol not available for: " + ', '.join(makams)
    except KeyError:
        print str(ii) + ' ' + key
        print "... No makam info available"

json.dump(anno_dict, open('../annotations.json', 'w'), indent=2)

