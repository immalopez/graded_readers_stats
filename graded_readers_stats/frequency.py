##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################

# word_counts[word][level] = [count, total]
# word_counts = {}
# for word in vocabulary:
#     for text_level_1 in texts_level_1:
#         pass
#     for text_level_2 in texts_level_2:
#         pass
#     for text_level_3 in texts_level_3:
#         pass

# "A Pepe le gusta comer HAMBURGUESAS en McDonald's cada miércoles."
#            ^     ^     ~~~~~~~~~~~~ ^  ^

# df['Occurrences_Reader_Level1'] = df.apply(
#   lambda x: frequency.count(x[LEMMA], readers_by_level['Level1']), axis=1
# )

# for word in vocabulary
#   for readers in level
#       frequency.count(word, readers)

# for readers in level
#   count(list of phrases, sentences)

import pandas as pd

# Typing
DataFrameGroupBy = pd.core.groupby.generic.DataFrameGroupBy
DataFrame = pd.DataFrame


def occurrences(items_df: DataFrame, content_df: DataFrameGroupBy):
    for group_name in content_df.groups:
        items_df[group_name] = items_df.apply(
            lambda x: count_occurrences(
                x['Lemma'],
                content_df.get_group(group_name)
            ),
            axis=1
        )


def count_occurrences(phrase: [[str]], texts: DataFrame):
    phrase, texts = phrase[0], texts['Lemma']

    count = 0
    for sents in texts:
        for sent in sents:
            if get_range_for_vocab_in_text(sent, phrase):
                count += 1
    return count


def get_range_for_vocab_in_text(text, vocabulary):
    """Returns a tuple(start, end) where start is the first index of
    vocabulary in text and end is the end index of the vocabulary in text if
    the lexical items contained in a vocabulary list are to be found in the
    sentence lists of a given text, and None otherwise."""
    t_pointer = 0
    v_pointer = 0
    len_text = len(text)
    len_vocab = len(vocabulary)
    while t_pointer < len_text and v_pointer < len_vocab:
        if str(text[t_pointer]).lower() == str(vocabulary[v_pointer]).lower():
            t_pointer += 1
            v_pointer += 1
            if v_pointer == len_vocab:
                # adjust start to include vocab item(s)
                return t_pointer - len_vocab, t_pointer  # a tuple
        else:
            t_pointer = t_pointer - v_pointer + 1
            v_pointer = 0
    return None
