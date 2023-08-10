cato_agri_praef = "Est interdum praestare mercaturis rem quaerere, nisi tam periculosum sit, et item foenerari, si tam honestum. Maiores nostri sic habuerunt et ita in legibus posiverunt: furem dupli condemnari, foeneratorem quadrupli. Quanto peiorem civem existimarint foeneratorem quam furem, hinc licet existimare. Et virum bonum quom laudabant, ita laudabant: bonum agricolam bonumque colonum; amplissime laudari existimabatur qui ita laudabatur. Mercatorem autem strenuum studiosumque rei quaerendae existimo, verum, ut supra dixi, periculosum et calamitosum. At ex agricolis et viri fortissimi et milites strenuissimi gignuntur, maximeque pius quaestus stabilissimusque consequitur minimeque invidiosus, minimeque male cogitantes sunt qui in eo studio occupati sunt. Nunc, ut ad rem redeam, quod promisi institutum principium hoc erit."

# first import a repo, the CLTK's data models for Latin
from cltk.corpus.utils.importer import CorpusImporter
corpus_importer = CorpusImporter('latin')
corpus_importer.import_corpus('latin_models_cltk')

# do j/v replace, then tokenize
from cltk.stem.latin.j_v import JVReplacer
from cltk.tokenize.word import WordTokenizer

jv_replacer = JVReplacer()
cato_agri_praef = jv_replacer.replace(cato_agri_praef.lower())

word_tokenizer = WordTokenizer('latin')
cato_word_tokens = word_tokenizer.tokenize(cato_agri_praef.lower())
cato_word_tokens = [token for token in cato_word_tokens if token not in ['.', ',', ':', ';']]

from cltk.stem.lemma import LemmaReplacer

lemmatizer = LemmaReplacer('latin')
lemmata = lemmatizer.lemmatize(cato_word_tokens)
print(lemmata)

# now do same but return original too
# useful for checking accuracy
lemmata_orig = lemmatizer.lemmatize(cato_word_tokens, return_raw=True)
print(lemmata_orig)

# now do counts again

# count all words
print(len(lemmata))  # 115

print(len(set(lemmata)))  # 73

# lexical diveristy, using lemmata
print(len(set(lemmata)) / len(lemmata))

