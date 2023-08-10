import json

def to_json(in_path, out_path):
    sentences = []
    with open(in_path, 'rt') as input_file:
        sentence_tokens = []
        sentence_tags = []
        for line in input_file:
            # Empty lines are sentence boundaries in the CoNLL format
            if line.strip() == '':
                if len(sentence_tokens) > 0:
                    sentences.append({'text': sentence_tokens, 'tags': sentence_tags})
                sentence_tokens = []
                sentence_tags = []
            elif '-DOCSTART' in line: # These are metadata lines we can skip
                continue
            else:
                data = line.strip().split()
                # Data format is <token> <pos> <chunk> <ner>
                sentence_tokens.append(data[0])
                sentence_tags.append(data[3].replace('I-MISC', 'O').replace('B-MISC', 'O').replace('B-', 'I-')) # Lets ignore MISC class
    with open(out_path, 'w') as output_file:
        json.dump(sentences, output_file, indent=2)

to_json('tagger/dataset/eng.train', 'data/ner_train.json')
to_json('tagger/dataset/eng.testa', 'data/ner_test.json')





