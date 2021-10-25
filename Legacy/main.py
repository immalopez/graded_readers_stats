###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################

from graded_readers_stats import preprocess, statistics, utils
# Possible alternatives for the Spanish native corpus:
# https://crscardellino.ar/SBWCE/ AND
# https://www.cs.upc.edu/~nlp/wikicorpus/
import pandas as pd

# ================================= LOAD DATA =================================

# Set to True to use a small sample of the whole dataset.
TRIAL = True

pd.set_option('display.max_columns', 99)

PATH_TO_READERS_CSV = '../Data/Graded readers list.csv'
PATH_TO_VOCABULARY_CSV = '../Data/Vocabulary list.csv'
PATH_TO_TRIAL_READERS_CSV = '../Data/Trial/Graded readers list.csv'
PATH_TO_TRIAL_VOCABULARY_CSV = '../Data/Trial/Vocabulary list.csv'

if TRIAL:
    graded_readers = pd.read_csv(PATH_TO_TRIAL_READERS_CSV, sep=';')
    graded_vocabulary = pd.read_csv(PATH_TO_TRIAL_VOCABULARY_CSV, sep=';')
    print("⚠️ WARNING: Using trial dataset!!!")
else:
    graded_readers = pd.read_csv(PATH_TO_READERS_CSV, sep=';')
    graded_vocabulary = pd.read_csv(PATH_TO_VOCABULARY_CSV, sep=';')

# ================================= CONSTANTS =================================

LEVEL_AVANZADO = 'Avanzado'
LEVEL_INTERMEDIO = 'Intermedio'
LEVEL_INICIAL = 'Inicial'
READER_LEVELS = [LEVEL_INICIAL, LEVEL_INTERMEDIO, LEVEL_AVANZADO]

# --- Column names --- #
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
VOCAB_FREQ_INICIAL = 'Vocab FreqDist for Level Inicial'
VOCAB_FREQ_INTERMEDIO = 'Vocab FreqDist for Level Intermedio'
VOCAB_FREQ_AVANZADO = 'Vocab FreqDist for Level Avanzado'

# ================================== COLUMNS ==================================


# --- GRADED READERS ---#

graded_readers[RAW_TEXT] = preprocess.read_files(
    graded_readers[TEXT_FILE]
)
graded_readers[PROCESSED_READERS] = preprocess.normalize_text(
    graded_readers[RAW_TEXT]
)
graded_readers[TOKENIZED_TXT] = graded_readers[PROCESSED_READERS].apply(
    preprocess.get_word_properties, args=('text',)
)
graded_readers[LEMMATIZED_TXT] = graded_readers[PROCESSED_READERS].apply(
    preprocess.get_word_properties, args=('lemma',)
)
graded_readers[TXT_UNI_POS] = graded_readers[PROCESSED_READERS].apply(
    preprocess.get_word_properties, args=('upos',)
)
graded_readers[TXT_FEATURES] = graded_readers[PROCESSED_READERS].apply(
    preprocess.get_word_properties, args=('feats',)
)
graded_readers[TXT_SYNTACTIC_HEAD] = graded_readers[PROCESSED_READERS].apply(
    preprocess.get_word_properties, args=('head',)
)
graded_readers[TXT_DEPENDENCY_RELATIONS] = graded_readers[
    PROCESSED_READERS].apply(
    preprocess.get_word_properties, args=('deprel',)
)
graded_readers[TXT_NAMED_ENTITIES] = graded_readers[PROCESSED_READERS].apply(
    preprocess.perform_ner
)

# --- GRADED VOCABULARY ---#

graded_vocabulary[PROCESSED_VOCAB] = preprocess.normalize_text(
    graded_vocabulary[LEXICAL_ITEM]
)
graded_vocabulary[TOKENIZED_VOCAB] = graded_vocabulary[PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('text',)
)
graded_vocabulary[LEMMATIZED_VOCAB] = graded_vocabulary[PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('lemma',)
)
graded_vocabulary[VOCAB_UNI_POS] = graded_vocabulary[PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('upos',)
)
graded_vocabulary[VOCAB_FEATURES] = graded_vocabulary[PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('feats',)
)
graded_vocabulary[VOCAB_SYNTACTIC_HEAD] = graded_vocabulary[
    PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('head',)
)
graded_vocabulary[VOCAB_DEPENDENCY_RELATIONS] = graded_vocabulary[
    PROCESSED_VOCAB].apply(
    preprocess.get_word_properties, args=('deprel',)
)
graded_vocabulary[VOCAB_NAMED_ENTITIES] = graded_vocabulary[
    PROCESSED_VOCAB].apply(
    preprocess.perform_ner
)
# graded_vocabulary[VOCAB_IN_READERS] = statistics.get_vocab_in_text(
#     graded_readers[LEMMATIZED_TXT], graded_vocabulary[LEMMATIZED_VOCAB]
# )

# --- FREQUENCY OF VOCABULARY ---#

vocab_counts_per_level = statistics.get_vocab_counts_in_texts(
    graded_vocabulary[LEMMATIZED_VOCAB],
    graded_readers[LEMMATIZED_TXT],
    graded_readers[LEVEL],
    READER_LEVELS
)

graded_vocabulary[VOCAB_FREQ_INICIAL] = graded_vocabulary.apply(
    lambda x: statistics.get_vocab_freq_dist_for_level(
        x[LEMMATIZED_VOCAB],
        LEVEL_INICIAL,
        vocab_counts_per_level
    ),
    axis=1
)
graded_vocabulary[VOCAB_FREQ_INTERMEDIO] = graded_vocabulary.apply(
    lambda x: statistics.get_vocab_freq_dist_for_level(
        x[LEMMATIZED_VOCAB],
        LEVEL_INTERMEDIO,
        vocab_counts_per_level
    ),
    axis=1
)
graded_vocabulary[VOCAB_FREQ_AVANZADO] = graded_vocabulary.apply(
    lambda x: statistics.get_vocab_freq_dist_for_level(
        x[LEMMATIZED_VOCAB],
        LEVEL_AVANZADO,
        vocab_counts_per_level
    ),
    axis=1
)


# --- FREQUENCY OF CONTEXT WORDS ---#


context_words_per_vocab_item = statistics.collect_vocab_context(
    graded_vocabulary[LEMMATIZED_VOCAB], graded_readers[LEMMATIZED_TXT])

graded_vocabulary['Context'] = graded_vocabulary.apply(
    # wrap into an empty list to match `get_vocab_in_texts` parameters later
    lambda x: context_words_per_vocab_item[
        utils.vocab_item_to_key(x[LEMMATIZED_VOCAB][0])
    ],
    axis=1
)


def get_vocab_context_counts(
        vocabulary: pd.Series,
        contexts: pd.Series,
        texts: pd.Series,
        levels: pd.Series,
        level_names: [str]):

    # result[word][word'][level] = (count, total)
    result = {}

    for vocab_items, context in zip(vocabulary, contexts):
        vocab_item = vocab_items[0]
        vocab_item_key = utils.vocab_item_to_key(vocab_item)
        result[vocab_item_key] = {}

        for word in context:
            result[vocab_item_key][word] = {
                level_names[0]: [0, 0],  # Level: [Count, Total]
                level_names[1]: [0, 0],  # Level: [Count, Total]
                level_names[2]: [0, 0]   # Level: [Count, Total]
            }

            for text_items, level in zip(texts, levels):
                for text_item in text_items:
                    # Increment total count
                    # WARNING: This counts sentences, not individual words!
                    result[vocab_item_key][word][level][1] += 1
                    # Increment occurrences
                    if statistics.get_range_for_vocab_in_text(text_item, [word]):
                        result[vocab_item_key][word][level][0] += 1
    return result


context_counts = get_vocab_context_counts(
    graded_vocabulary[LEMMATIZED_VOCAB],
    graded_vocabulary['Context'],
    graded_readers[LEMMATIZED_TXT],
    graded_readers[LEVEL],
    READER_LEVELS
)


def aggregate_context_counts_per_level(context_counts, level_names):
    level1 = level_names[0]
    level2 = level_names[1]
    level3 = level_names[2]
    # result[vocab_item][level] = [count, total]
    result = {}
    for vocab_key, context_dict in context_counts.items():
        result[vocab_key] = {
            level_names[0]: [0, 0],  # Level: [Count, Total]
            level_names[1]: [0, 0],  # Level: [Count, Total]
            level_names[2]: [0, 0]   # Level: [Count, Total]
        }
        for word, levels_dict in context_dict.items():
            result[vocab_key][level1][0] += levels_dict[level1][0]
            result[vocab_key][level1][1] += levels_dict[level1][1]
            result[vocab_key][level2][0] += levels_dict[level2][0]
            result[vocab_key][level2][1] += levels_dict[level2][1]
            result[vocab_key][level3][0] += levels_dict[level3][0]
            result[vocab_key][level3][1] += levels_dict[level3][1]
            # for level_name, counts in levels_dict.items():
            #     result[vocab_key][level_name][0] += counts[0]
            #     result[vocab_key][level_name][1] += counts[1]
    return result


aggregates = aggregate_context_counts_per_level(context_counts, READER_LEVELS)
print(aggregates)
# graded_vocabulary['CTX_FREQ_INICIAL'] = graded_vocabulary.apply(
#     lambda x: get_vocab_context_counts(
#         x['Context'],
#         graded_readers[LEMMATIZED_TXT],
#         graded_readers[LEVEL],
#         READER_LEVELS
#     ),
#     axis=1
# )
# graded_vocabulary['CTX_FREQ_INTERMEDIO'] = graded_vocabulary.apply(
#     lambda x: utils.get_vocab_freq_for_level(
#         x['Context'],
#         LEVEL_INTERMEDIO,
#         context_counts_per_level
#     ),
#     axis=1
# )
# graded_vocabulary['CTX_FREQ_AVANZADO'] = graded_vocabulary.apply(
#     lambda x: utils.get_vocab_freq_for_level(
#         x['Context'],
#         LEVEL_AVANZADO,
#         context_counts_per_level
#     ),
#     axis=1
# )

# print(context_counts)

# def get_vocab_counts_in_texts(vocabulary: pd.Series,
#                               texts: pd.Series,
#                               levels: pd.Series,
#                               level_names: [str]):
#     """Counts how many occurrences of each vocabulary item can be found in the
#     texts of each language level."""
#     counts_per_level = {}
#     for vocab_items in vocabulary:
#         vocab_item = vocab_items[0]
#         vocab_item_key = vocab_item_to_key(vocab_item)
#
#         counts_per_level[vocab_item_key] = {
#             level_names[0]: [0, 0],  # Level 1: [Occurrences, Total]
#             level_names[1]: [0, 0],  # Level 2: [Occurrences, Total]
#             level_names[2]: [0, 0]   # Level 3: [Occurrences, Total]
#         }
#         for text_items, level in zip(texts, levels):
#             for text_item in text_items:
#
#                 # Increment total count
#                 # WARNING: This counts sentences, not individual words!
#                 counts_per_level[vocab_item_key][level][1] += 1
#
#                 # Increment occurrences
#                 if get_range_for_vocab_in_text(text_item, vocab_item):
#                     counts_per_level[vocab_item_key][level][0] += 1
#
#     return counts_per_level


# adapted_context_words = [[[context_word]] for context_word in context_words]
# word_counts_per_level = statistics.get_vocab_counts_in_texts(
#     adapted_context_words,
#     graded_readers[LEMMATIZED_TXT],
#     graded_readers[LEVEL],
#     READER_LEVELS
# )
# print(word_counts_per_level)


# get_vocab_context_counts(['word1', 'word2'], None, None, None)


# context_counts_per_level = statistics.get_vocab_in_texts_freq(
#     graded_vocabulary['Context'],
#     graded_readers[LEMMATIZED_TXT],
#     graded_readers[LEVEL],
#     READER_LEVELS
# )



# --- LITERATURE PARTITIONS ---#

# --- NATIVE CORPUS ---#


# ================================ CLEAN DATA ================================

if __name__ == '__main__':
    print(graded_readers[LEMMATIZED_TXT])
    print(graded_vocabulary[LEMMATIZED_VOCAB])
    print(statistics.get_vocab_in_text(graded_readers[LEMMATIZED_TXT],
                                       graded_vocabulary[LEMMATIZED_VOCAB]))
