##############################################################################
#                                DATA LOADING                                #
##############################################################################
# NOTE:
# Possible alternatives for the Spanish native corpus:
# https://crscardellino.ar/SBWCE/ AND
# https://www.cs.upc.edu/~nlp/wikicorpus/

import os
from enum import Enum
from collections import namedtuple
from typing import Optional

import pandas as pd

from graded_readers_stats.constants import COL_RAW_TEXT

pd.set_option('display.max_columns', 99)

FOLDER_DATA = './Data/'
FOLDER_DATA_TRIAL = './Data_trial/'


class Dataset(Enum):
    VOCABULARY = 'Vocabulary'
    READERS = 'Readers'
    LITERATURE = 'Literature'
    NATIVE = 'Native'


DatasetInfo = namedtuple('DatasetInfo', ['csv_file', 'cache_file'])

datasets = {
    Dataset.VOCABULARY: DatasetInfo(
        csv_file='Vocabulary list.csv',
        cache_file='cache/vocabulary.pkl',
    ),
    Dataset.READERS: DatasetInfo(
        csv_file='Graded readers list.csv',
        cache_file='cache/readers.pkl',
    ),
    Dataset.LITERATURE: DatasetInfo(
        csv_file='Literature list.csv',
        cache_file='cache/literature.pkl',
    ),
    Dataset.NATIVE: DatasetInfo(
        csv_file='Native list.csv',
        cache_file='cache/native.pkl',
    ),
}


# ================================= LOAD DATA =================================
is_trial = False


def load(
        dataset: Dataset,
        trial: bool = False,
        use_cache: bool = True,
        folder: Optional[str] = None
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    global is_trial
    print(f'Loading {dataset.name}, use_cache = {str(use_cache)}:')
    if trial:
        print("\t⚠️ WARNING: Using trial dataset!!!")

    if trial:
        folder = FOLDER_DATA_TRIAL if folder is None else folder
        is_trial = True
    else:
        folder = FOLDER_DATA if folder is None else folder

    csv_file = folder + datasets[dataset].csv_file
    cache_file = folder + datasets[dataset].cache_file
    load_file = load_native_corpus \
        if dataset == Dataset.NATIVE \
        else read_pandas_csv

    if use_cache:
        dataframe = pd.read_pickle(cache_file) \
            if os.path.exists(cache_file) \
            else load_file(csv_file)
    else:
        dataframe = load_file(csv_file)

    print(f'\t{dataset.name} loaded!')
    return dataframe


def read_pandas_csv(path: str) -> pd.DataFrame:
    print(f'\tReading csv file at {path}')
    return pd.read_csv(path, sep=';')


def load_native_corpus(*args) -> pd.DataFrame:
    """Loads the native corpus from the NLTK library."""
    from nltk.corpus import cess_esp
    words_esp = cess_esp.words()
    if is_trial:
        words_esp = [
            'Word1',
            'earth',
            'word2',
            'summer',
            '-Fpa-',
            'BetweenParenthesis',
            '-Fpt-',
            '-fe-',
            '*0*',
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

    # Sanitize metadata
    import re
    native = re.sub(r'-[Ff]pa-', '(', native)
    native = re.sub(r'-[Ff]pt-', ')', native)
    native = re.sub(r'\*0\*', '', native)
    native = re.sub(r'-[Ff]e-', '', native)

    dataframe = pd.DataFrame({COL_RAW_TEXT: [native], 'Level': ['Native']})
    return dataframe


def save(
        vocabulary: pd.DataFrame,
        readers: pd.DataFrame,
        literature: pd.DataFrame,
        native: pd.DataFrame,
        folder: Optional[str] = None
) -> None:

    if is_trial:
        folder = FOLDER_DATA_TRIAL if folder is None else folder
        print("Saving data (TRIAL)")
    else:
        folder = FOLDER_DATA if folder is None else folder
        print("Saving data (REAL)")

    vocabulary.to_pickle(folder + datasets[Dataset.VOCABULARY].cache_file)
    readers.to_pickle(folder + datasets[Dataset.READERS].cache_file)
    literature.to_pickle(folder + datasets[Dataset.LITERATURE].cache_file)
    native.to_pickle(folder + datasets[Dataset.NATIVE].cache_file)


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
