##############################################################################
#                            CORPUS PREPROCESSING                            #
##############################################################################

# Using Stanford NLP's "Stanza" package.
# Documentation available at: https://stanfordnlp.github.io/stanza/
import stanza as st

# ================================= CONSTANTS =================================

nlp_es = st.Pipeline(
    lang='es',
    processors='tokenize,mwt,pos,lemma,depparse,ner'
)

# --- Column names --- #
LEXICAL_ITEM = 'Lexical item'
LEVEL = 'Level'
TEXT_FILE = 'Text file'
RAW_TEXT = "Raw text"

NORMALIZED_TEXT = 'Normalized text'
TOKENIZED_TEXT = 'Tokenized text'
LEMMATIZED_TEXT = 'Lemmatized text'

NORMALIZED_VOCAB = 'Normalized vocab'
TOKENIZED_VOCAB = 'Tokenized vocab'
LEMMATIZED_VOCAB = 'Lemmatized vocab'

# LEVEL_AVANZADO = 'Avanzado'
# LEVEL_INTERMEDIO = 'Intermedio'
# LEVEL_INICIAL = 'Inicial'
# READER_LEVELS = [LEVEL_INICIAL, LEVEL_INTERMEDIO, LEVEL_AVANZADO]


# ============================== PREPROCESS DATA ==============================


def get_word_properties(document, key):
    """Returns a list of lists (sentences) for each document with the specified
    property (id, text, lemma, upos, xpos, feats, start_char, & end_char)."""
    all_sentences = []
    for sentence in document.to_dict():
        sentence_result = []
        for word in sentence:
            sentence_result.append(word.get(key))
        all_sentences.append(sentence_result)
    return all_sentences


def get_fields(document, key):
    """Returns a list of lists (sentences) for each document with the specified
    property (id, text, lemma, upos, xpos, feats, start_char, & end_char)."""
    return document.get(key, True)


def normalize_text(texts):
    """Processes a list of strings and returns it as a list of lists (docs)
    of sublists, wherein each sublist is a sentence that in turn contains a
    dictionary per word with the following information: "id" (ID of the word),
    "text" (token), "lemma" (lemmatized token), "upos" (universal POS), "xpos"
    (treebank-specific POS), "feats" (universal morphological features of the
    word), "start_char" (starting character) & "end_char" (ending one)."""
    return [nlp_es(text) for text in texts]


def perform_ner(document):
    """Returns a list for each document with any detected named entity
    types."""
    return [token.ner
            for sentence in document.sentences
            for token in sentence.tokens]

text_analysis_pipeline = [
    preprocess.normalize_text,
    preprocess.tokenize_text,
    # preprocess.uni_pos,
    # preprocess.features,
    # preprocess.syntactic_head,
    # preprocess.dep_relations,
    # preprocess.named_entities
]
vocabulary_pipeline = [
    preprocess.normalize_text,
    preprocess.tokenize_text
]

# ================================== COLUMNS ==================================

# graded_readers[RAW_TEXT] = preprocess.read_files(
#     graded_readers[TEXT_FILE]
# )
# graded_readers[NORMALIZED_TEXT] = preprocess.normalize_text(
#     graded_readers[RAW_TEXT]
# )
# graded_readers[TOKENIZED_TEXT] = graded_readers[NORMALIZED_TEXT].apply(
#     preprocess.get_fields, args=('text',)
# )
# graded_readers[LEMMATIZED_TEXT] = graded_readers[NORMALIZED_TEXT].apply(
#     preprocess.get_fields, args=('lemma',)
# )
# graded_readers['blah'] = graded_readers[NORMALIZED_TEXT].apply(
#     preprocess.get_word_properties, args=('lemma',)
# )
