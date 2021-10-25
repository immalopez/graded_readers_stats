##############################################################################
#                            CORPUS PREPROCESSING                            #
##############################################################################

# Using Stanford NLP's "Stanza" package.
# Documentation available at: https://stanfordnlp.github.io/stanza/
import stanza as st


# ================================= CONSTANTS =================================

nlp_es = st.Pipeline(lang='es',
                     processors='tokenize,mwt,pos,lemma,depparse,ner')


# ================================= LOAD DATA =================================

def read_files(paths):
    """Opens each text file in a list and returns a single list of strings with
    all of their contents."""
    texts = []
    for path in paths:
        with open(path) as file:
            text = file.read()
            texts.append(text)
    return texts


# ============================== PREPROCESS DATA ==============================

def normalize_text(texts):
    """Processes a list of strings and returns it as a list of lists (docs)
    of sublists, wherein each sublist is a sentence that in turn contains a
    dictionary per word with the following information: "id" (ID of the word),
    "text" (token), "lemma" (lemmatized token), "upos" (universal POS), "xpos"
    (treebank-specific POS), "feats" (universal morphological features of the
    word), "start_char" (starting character) & "end_char" (ending one)."""
    return [nlp_es(text) for text in texts]


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


def perform_ner(document):
    """Returns a list for each document with any detected named entity
    types."""
    return [token.ner
            for sentence in document.sentences
            for token in sentence.tokens]
