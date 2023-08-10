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
            elif line.startswith('#'): # These are metadata lines we can skip
                continue
            else:
                data = line.strip().split("\t")
                if "-" in data[0]: #token range line, skip
                    continue
                sentence_tokens.append(data[1])
                sentence_tags.append(data[3])
    with open(out_path, 'w') as output_file:
        json.dump(sentences, output_file, indent=2)

to_json('UD_English-EWT/en-ud-train.conllu', 'data/pos_train.json')
to_json('UD_English-EWT/en-ud-dev.conllu', 'data/pos_devel.json')
to_json('UD_Finnish-TDT/fi-ud-train.conllu', 'data/pos_train_fi.json')
to_json('UD_Finnish-TDT/fi-ud-dev.conllu', 'data/pos_devel_fi.json')

