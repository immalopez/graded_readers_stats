##############################################################################
#                            CORPUS PREPROCESSING                            #
##############################################################################

import os

import stanza as st

from graded_readers_stats import data
from graded_readers_stats.utils import *
from graded_readers_stats._typing import *
from graded_readers_stats.constants import *

# ================================= CONSTANTS =================================

nlp_es = st.Pipeline(
    lang='es',
    processors='tokenize,mwt,lemma,pos,depparse',
    logging_level='ERROR'
)

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


def get_fields(documents, key):
    """Returns a list of lists (sentences) for each document with the specified
    property (id, text, lemma, upos, xpos, feats, start_char, & end_char)."""
    return [d.get(key, True) for d in documents]


def collect_context_for_phrases_in_texts(
        phrases: DataFrame,
        texts: DataFrame,
        column_prefix: str
) -> DataFrame:
    phrases[column_prefix + SUFFIX_CONTEXT] = phrases.apply(
        lambda x: collect_context_for_phrase_in_texts(
            x[COL_LEMMA][0],
            texts[COL_LEMMA]
        ),
        axis=1
    )
    return phrases


def collect_context_for_phrase_in_texts(
        phrase: [str],
        texts: Series,
        window: int = 3
) -> [str]:
    context = []
    for sentences in texts:
        for sent in sentences:
            text_range = get_range_of_phrase_in_sentence(phrase, sent)
            if text_range:
                start, end = text_range[0], text_range[1]

                # limit indices to sentence bounds using `min` and `max`
                # to safely use with list slicing
                # since out-of-bounds slicing would return empty list ([])
                slice_before = slice(max(0, start - window), start)
                slice_after = slice(end, min(end + window, len(sent)))

                words = sent[slice_before] + sent[slice_after]
                context.extend(words)

    return context


def process_and_store_stanza_doc(stanza_path):
    print('Processing: ' + stanza_path)
    path_no_ext = os.path.splitext(stanza_path)[0]
    text = data.read_files([path_no_ext])[0]
    doc = nlp_es(text)
    serialized = doc.to_serialized()
    with open(stanza_path, 'wb') as f:
        f.write(serialized)


def restore_stanza_file(path):
    with open(path + '.stanza', 'rb') as f:
        serialized = f.read()
        return st.Document.from_serialized(serialized)


def process_or_restore_stanza_docs(df: DataFrame) -> DataFrame:
    file_paths = df[COL_TEXT_FILE]

    print('Processing stanza docs...')
    for path in file_paths:
        stanza_path = path + '.stanza'
        if not os.path.exists(stanza_path):
            process_and_store_stanza_doc(stanza_path)
    print('Processing stanza docs DONE!')

    df[COL_STANZA_DOC] = df[COL_TEXT_FILE].apply(restore_stanza_file)

    return df


# =========================== PIPELINE EXECUTION ==============================
# Pipeline steps receive a DataFrame, transform it, and return it for next step


text_analysis_pipeline = [
    set_column(read_files, src=COL_TEXT_FILE, dst=COL_RAW_TEXT),
    # process_or_restore_stanza_docs,
    set_column(normalize_text, src=COL_RAW_TEXT, dst=COL_STANZA_DOC),
    set_column(get_fields, src=COL_STANZA_DOC, dst=COL_LEMMA, args=('lemma',)),
]
vocabulary_pipeline = [
    # process_or_restore_stanza_docs,
    set_column(normalize_text, src=COL_LEXICAL_ITEM, dst=COL_STANZA_DOC),
    set_column(get_fields, src=COL_STANZA_DOC, dst=COL_LEMMA, args=('lemma',)),
]


def run(df: DataFrame, pipes: [Pipe]) -> DataFrame:
    for pipe in pipes:
        df = pipe(df)
    return df


