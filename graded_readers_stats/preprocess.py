##############################################################################
#                            CORPUS PREPROCESSING                            #
##############################################################################

from collections.abc import Callable

import stanza as st
import pandas as pd

from graded_readers_stats import data

# ================================= CONSTANTS =================================

nlp_es = st.Pipeline(
    lang='es',
    processors='tokenize,mwt,lemma'  # less processors for faster execution
    # processors='tokenize,mwt,pos,lemma,depparse,ner'
)

# --- Column names --- #
LEXICAL_ITEM = 'Lexical item'
LEVEL = 'Level'
TEXT_FILE = 'Text file'
RAW_TEXT = 'Raw text'
STANZA_DOC = 'Stanza Doc'
LEMMA = 'Lemma'

# Type hints
DataFrame = pd.DataFrame
Pipe = Callable[[DataFrame], DataFrame]

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


def get_fields(documents, key):
    """Returns a list of lists (sentences) for each document with the specified
    property (id, text, lemma, upos, xpos, feats, start_char, & end_char)."""
    return [d.get(key, True) for d in documents]


# ============================== PIPELINE STEPS ===============================


def set_column(func, src, dst, args=()):
    """
    Returns a func to be used in a pipeline (i.e. a func that receives and
    returns a DataFrame). Parameters are used to specify on what
    data to work on and which transformation to apply.

    :param dst: Column name to assign in the modified DataFrame.
    :param src: Column name from which to take the data.
    :param func: Transformer func working with the data.
    :param args: Any extra args the transformer might need for its work.
    :return: A preconfigured func with input and output columns and an
    operation to run on the data. The returned func is intended to be used
    in our pipeline (i.e. receives and returns a DatFrame).
    """
    def modify(df):
        df[dst] = func(df[src], *args)
        return df
    return modify


def read_files(file_paths: [str]) -> [str]:
    """
    Read files from the file system. Delegates actual work to `data` module.

    :param file_paths: Paths to open for reading.
    :return: a list of strings with content of the input files.
    """
    return data.read_files(file_paths)


def normalize_text(texts):
    """Processes a list of strings and returns it as a list of lists (docs)
    of sublists, wherein each sublist is a sentence that in turn contains a
    dictionary per word with the following information: "id" (ID of the word),
    "text" (token), "lemma" (lemmatized token), "upos" (universal POS), "xpos"
    (treebank-specific POS), "feats" (universal morphological features of the
    word), "start_char" (starting character) & "end_char" (ending one)."""

    # An inefficient way to process texts one by one.
    # Left here for reference.
    # return [nlp_es(text) for text in texts]

    # Wrap each document with a stanza.Document object
    documents = [st.Document([], text=d) for d in texts]
    # Call the neural pipeline on this list of documents
    return nlp_es(documents)


# =========================== PIPELINE EXECUTION ==============================
# Pipeline steps receive a DataFrame, transform it, and return it for next step


text_analysis_pipeline = [
    set_column(read_files, src=TEXT_FILE, dst=RAW_TEXT),
    set_column(normalize_text, src=RAW_TEXT, dst=STANZA_DOC),
    set_column(get_fields, src=STANZA_DOC, dst=LEMMA, args=('lemma',)),
]
vocabulary_pipeline = [
    set_column(normalize_text, src=LEXICAL_ITEM, dst=STANZA_DOC),
    set_column(get_fields, src=STANZA_DOC, dst=LEMMA, args=('lemma',)),
]


def run(df: DataFrame, pipes: [Pipe]) -> DataFrame:
    for pipe in pipes:
        df = pipe(df)
    return df
