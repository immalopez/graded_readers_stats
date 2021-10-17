###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################

import pandas as pd

# Possible alternatives for the Spanish native corpus:
# https://crscardellino.ar/SBWCE/ AND
# https://www.cs.upc.edu/~nlp/wikicorpus/
from nltk.corpus import cess_esp as cess
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from graded_readers_stats import preprocess, statistics


# ================================= LOAD DATA =================================

# Set to True to use a small sample of the whole dataset.
TRIAL = True

pd.set_option('display.max_columns', 99)

PATH_TO_READERS_CSV = 'Data/Graded readers list.csv'
PATH_TO_VOCABULARY_CSV = 'Data/Vocabulary list.csv'
PATH_TO_TRIAL_READERS_CSV = 'Data/Trial/Graded readers list.csv'
PATH_TO_TRIAL_VOCABULARY_CSV = 'Data/Trial/Vocabulary list.csv'

if TRIAL:
    graded_readers = pd.read_csv(PATH_TO_TRIAL_READERS_CSV, sep=';')
    graded_vocabulary = pd.read_csv(PATH_TO_TRIAL_VOCABULARY_CSV, sep=';')
    print("⚠️ WARNING: Using trial dataset!!!")
else:
    graded_readers = pd.read_csv(PATH_TO_READERS_CSV, sep=';')
    graded_vocabulary = pd.read_csv(PATH_TO_VOCABULARY_CSV, sep=';')


# ================================= CONSTANTS =================================

# --- Column names ---#
LEXICAL_ITEM = 'Lexical item'
RAW_TEXT = "Reader's raw text"
LEVEL = 'Level'
TEXT_FILE = 'Text file'
PROCESSED_READERS = 'Processed readers'
PROCESSED_VOCAB = 'Processed lexical items'
TOKENIZED_TXT = 'Tokenized text'
TOKENIZED_VOCAB = 'Tokenized vocabulary'
LEMMATIZED_TXT = 'Lemmatized text'
LEMMATIZED_VOCAB = 'Lemmatized vocabulary'
TXT_UNI_POS = "Text's universal POS"
VOCAB_UNI_POS = "Vocabulary's universal POS"
TXT_FEATURES = "Text's universal morphological features"
VOCAB_FEATURES = "Vocabulary's universal morphological features"
TXT_SYNTACTIC_HEAD = "Text's syntactic heads"
VOCAB_SYNTACTIC_HEAD = "Vocabulary's syntactic heads"
TXT_DEPENDENCY_RELATIONS = "Text's dependency relations"
VOCAB_DEPENDENCY_RELATIONS = "Vocabulary's dependency relations"
TXT_NAMED_ENTITIES = "Text's named entities"
VOCAB_NAMED_ENTITIES = "Vocabulary's named entities"
VOCAB_IN_READERS = 'Is this vocabulary item in the readers?'


# ================================== COLUMNS ==================================

graded_readers[RAW_TEXT] = preprocess.read_readers(
    graded_readers[TEXT_FILE]
)
graded_readers[PROCESSED_READERS] = preprocess.normalize_text(
    graded_readers[RAW_TEXT]
)
graded_vocabulary[PROCESSED_VOCAB] = preprocess.normalize_text(
    graded_vocabulary[LEXICAL_ITEM]
)
graded_readers[TOKENIZED_TXT] = graded_readers[PROCESSED_READERS].apply(
    preprocess.get_word_properties, args=('text',)
)
graded_vocabulary[TOKENIZED_VOCAB] = graded_vocabulary[PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('text',)
)
graded_readers[LEMMATIZED_TXT] = graded_readers[PROCESSED_READERS].apply(
    preprocess.get_word_properties, args=('lemma',)
)
graded_vocabulary[LEMMATIZED_VOCAB] = graded_vocabulary[PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('lemma',)
)
graded_readers[TXT_UNI_POS] = graded_readers[PROCESSED_READERS].apply(
    preprocess.get_word_properties, args=('upos',)
)
graded_vocabulary[VOCAB_UNI_POS] = graded_vocabulary[PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('upos',)
)
graded_readers[TXT_FEATURES] = graded_readers[PROCESSED_READERS].apply(
    preprocess.get_word_properties, args=('feats',)
)
graded_vocabulary[VOCAB_FEATURES] = graded_vocabulary[PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('feats',)
)
graded_readers[TXT_SYNTACTIC_HEAD] = graded_readers[PROCESSED_READERS].apply(
    preprocess.get_word_properties, args=('head',)
)
graded_vocabulary[VOCAB_SYNTACTIC_HEAD] = graded_vocabulary[
    PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('head',)
)
graded_readers[TXT_DEPENDENCY_RELATIONS] = graded_readers[
    PROCESSED_READERS].apply(
    preprocess.get_word_properties, args=('deprel',)
)
graded_vocabulary[VOCAB_DEPENDENCY_RELATIONS] = graded_vocabulary[
    PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('deprel',)
)
graded_readers[TXT_NAMED_ENTITIES] = graded_readers[PROCESSED_READERS].apply(
    preprocess.perform_ner
)
graded_vocabulary[VOCAB_NAMED_ENTITIES] = graded_vocabulary[
    PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('text',)
)
graded_vocabulary[VOCAB_IN_READERS] = statistics.get_vocab_in_text(
    graded_readers[LEMMATIZED_TXT], graded_vocabulary[LEMMATIZED_VOCAB]
)


def count_vocab_in_texts(vocab: pd.Series, texts: pd.Series, levels: pd.Series):
    counts_per_level = {}
    for vocab_items in vocab:
        vocab_item = vocab_items[0]
        vocab_item_key = '_'.join(vocab_item)
        counts_per_level[vocab_item_key] = {
            'Inicial': 0,
            'Intermedio': 0,
            'Avanzado': 0
        }
        for text_items, level in zip(texts, levels):
            for text_item in text_items:
                if statistics.check_if_vocab_in_text(text_item, vocab_item):
                    counts_per_level[vocab_item_key][level] += 1
    return counts_per_level


vocab_counts_per_level = count_vocab_in_texts(
    graded_vocabulary[LEMMATIZED_VOCAB],
    graded_readers[LEMMATIZED_TXT],
    graded_readers[LEVEL]
)
print(vocab_counts_per_level)


def temp(vocab_item, level):
    key = '_'.join(vocab_item[0])
    return vocab_counts_per_level[key][level]


graded_vocabulary['Inicial'] = graded_vocabulary.apply(
    lambda x: temp(x[LEMMATIZED_VOCAB], 'Inicial'),
    axis=1
)
graded_vocabulary['Intermedio'] = graded_vocabulary.apply(
    lambda x: temp(x[LEMMATIZED_VOCAB], 'Intermedio'),
    axis=1
)
graded_vocabulary['Avanzado'] = graded_vocabulary.apply(
    lambda x: temp(x[LEMMATIZED_VOCAB], 'Avanzado'),
    axis=1
)

# ================================ CLEAN DATA ================================

if __name__ == '__main__':
    print(graded_readers[LEMMATIZED_TXT])
    print(graded_vocabulary[LEMMATIZED_VOCAB])
    print(statistics.get_vocab_in_text(graded_readers[LEMMATIZED_TXT],
                                       graded_vocabulary[LEMMATIZED_VOCAB]))
#    print(statistics.compute_vocab_freq_dist(
#        graded_vocabulary[LEMMATIZED_VOCAB],
#        graded_vocabulary[VOCAB_IN_READERS],
#        graded_readers[LEMMATIZED_TXT]))
