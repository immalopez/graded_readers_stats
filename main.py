###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################

import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from graded_readers_stats import preprocess, statistics


# ================================= LOAD DATA =================================

pd.set_option('display.max_columns', 99)

PATH_TO_READERS_CSV = 'Data/Graded readers list.csv'
PATH_TO_VOCABULARY_CSV = 'Data/Vocabulary list.csv'

graded_readers = pd.read_csv(PATH_TO_READERS_CSV, sep=';')[0:2] # REMEMBER TO CHANGE BACK
graded_vocabulary = pd.read_csv(PATH_TO_VOCABULARY_CSV, sep=';')[0:2] # REMEMBER TO CHANGE BACK


# ================================= CONSTANTS =================================

# --- Column names ---#
LEXICAL_ITEM = 'Lexical item'
TEXT = "Reader's text"
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
VOCAB_NAMED_ENTITIES = "Vocabulary's named entitites"


# ================================== COLUMNS ==================================

graded_readers[TEXT] = preprocess.read_readers(
    graded_readers[TEXT_FILE]
)
graded_readers[PROCESSED_READERS] = preprocess.normalize_text(
    graded_readers[TEXT]
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


# ================================ CLEAN DATA ================================

if __name__ == '__main__':
    print(graded_vocabulary[VOCAB_UNI_POS])
    print(graded_vocabulary[LEMMATIZED_VOCAB])
    print(graded_vocabulary[VOCAB_SYNTACTIC_HEAD])
    print(graded_vocabulary[VOCAB_DEPENDENCY_RELATIONS])
    print(graded_vocabulary[VOCAB_NAMED_ENTITIES])
    print(statistics.compute_freq_dist(graded_readers[LEMMATIZED_TXT],
                                       graded_vocabulary[LEMMATIZED_VOCAB]))
