##############################################################################
#                                DATA LOADING                                #
##############################################################################
# NOTE:
# Possible alternatives for the Spanish native corpus:
# https://crscardellino.ar/SBWCE/ AND
# https://www.cs.upc.edu/~nlp/wikicorpus/

import os
from typing import Optional

import pandas as pd

from graded_readers_stats.constants import COL_RAW_TEXT

pd.set_option('display.max_columns', 99)

FOLDER_DATA = './Data/'
FOLDER_DATA_TRIAL = './Data_trial/'

VOCABULARY_CSV = 'Vocabulary list.csv'
READERS_CSV = 'Graded readers list.csv'
LITERATURE_CSV = 'Literature list.csv'
NATIVE_CSV = 'Native list.csv'

VOCABULARY_CACHE = 'cache/vocabulary.pkl'
READERS_CACHE = 'cache/readers.pkl'
LITERATURE_CACHE = 'cache/literature.pkl'
NATIVE_CACHE = 'cache/native.pkl'


# ================================= LOAD DATA =================================
is_trial = False


def load(
        trial: bool = False,
        use_cache: bool = True,
        folder: str = Optional[str]
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    global is_trial
    print('Loading data, use_cache = ' + str(use_cache))

    if trial:
        folder = FOLDER_DATA_TRIAL if folder is None else folder
        is_trial = True
        print("⚠️ WARNING: Using trial dataset!!!")
    else:
        folder = FOLDER_DATA if folder is None else folder

    if use_cache:
        vocabulary = pd.read_pickle(folder + VOCABULARY_CACHE) \
            if os.path.exists(folder + VOCABULARY_CACHE) \
            else pd.read_csv(folder + VOCABULARY_CSV, sep=';')
        readers = pd.read_pickle(folder + READERS_CACHE) \
            if os.path.exists(folder + READERS_CACHE) \
            else pd.read_csv(folder + READERS_CSV, sep=';')
        literature = pd.read_pickle(folder + LITERATURE_CACHE) \
            if os.path.exists(folder + LITERATURE_CACHE) \
            else pd.read_csv(folder + LITERATURE_CSV, sep=';')
        native = pd.read_pickle(folder + NATIVE_CACHE) \
            if os.path.exists(folder + NATIVE_CACHE) \
            else load_native_corpus()

    else:
        vocabulary = pd.read_csv(folder + VOCABULARY_CSV, sep=';')
        readers = pd.read_csv(folder + READERS_CSV, sep=';')
        literature = pd.read_csv(folder + LITERATURE_CSV, sep=';')
        native = load_native_corpus()

    return vocabulary, readers, literature , native


def load_native_corpus() -> pd.DataFrame:
    """Loads the native corpus from the NLTK library."""
    from nltk.corpus import cess_esp
    words_esp = cess_esp.words()
    if is_trial:
        words_esp = [
            'Word1',
            'earth',
            'word2',
            'summer',
            'word3',
            'sun',
            ' ',
            '.',
            ' ',
            'Dawn',
            'this',
            'is',
            'it',
            'word5',
            'peter',
            'pan',
            'multi word 1',
            'morning',
        ]
    native = ' '.join([w.replace('_', ' ') for w in words_esp])
    dataframe = pd.DataFrame({COL_RAW_TEXT: [native], 'Level': ['Native']})
    return dataframe


def save(
        vocabulary: pd.DataFrame,
        readers: pd.DataFrame,
        literature: pd.DataFrame
) -> None:

    if is_trial:
        folder = FOLDER_DATA_TRIAL
        print("Saving data (TRIAL)")
    else:
        folder = FOLDER_DATA
        print("Saving data (REAL)")

    vocabulary.to_pickle(folder + VOCABULARY_CACHE)
    readers.to_pickle(folder + READERS_CACHE)
    literature.to_pickle(folder + LITERATURE_CACHE)

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
