##############################################################################
#                                DATA LOADING                                #
##############################################################################
import pandas as pd
from graded_readers_stats import preprocess

pd.set_option('display.max_columns', 99)

PATH_TO_READERS_CSV = './Data/Graded readers list.csv'
PATH_TO_VOCABULARY_CSV = './Data/Vocabulary list.csv'
PATH_TO_TRIAL_READERS_CSV = './Data/Trial/Graded readers list.csv'
PATH_TO_TRIAL_VOCABULARY_CSV = './Data/Trial/Vocabulary list.csv'


# ================================= LOAD DATA =================================


def load(trial: bool = False) -> (pd.DataFrame, pd.DataFrame):
    if trial:
        print("⚠️ WARNING: Using trial dataset!!!")
        graded_readers = pd.read_csv(PATH_TO_TRIAL_READERS_CSV, sep=';')
        graded_vocabulary = pd.read_csv(PATH_TO_TRIAL_VOCABULARY_CSV, sep=';')
    else:
        graded_readers = pd.read_csv(PATH_TO_READERS_CSV, sep=';')
        graded_vocabulary = pd.read_csv(PATH_TO_VOCABULARY_CSV, sep=';')
    return graded_readers, graded_vocabulary


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
