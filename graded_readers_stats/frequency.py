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


def count_phrases_in_sentences_by_groups(
        phrases_dataframe: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        new_column_prefix: str
) -> None:
    for group_name in sentences_by_groups.groups:
        output_column_name = new_column_prefix + group_name
        phrases_dataframe[output_column_name] = phrases_dataframe.apply(
            lambda x: phrase_count_in_sents(
                x['Lemma'],
                sentences_by_groups.get_group(group_name)
            ),
            axis=1
        )


def phrase_count_in_sents(phrase: [[str]], texts: DataFrame) -> int:
    phrase, texts = phrase[0], texts['Lemma']

    count = 0
    for sents in texts:
        for sent in sents:
            if get_range_of_phrase_in_sentence(phrase, sent):
                count += 1
    return count


def get_range_of_phrase_in_sentence(phrase: [str], sentence: [str]):
    """Returns a tuple(start, end) where start is the first index of
    vocabulary in text and end is the end index of the vocabulary in text if
    the lexical items contained in a vocabulary list are to be found in the
    sentence lists of a given text, and None otherwise."""
    sent_index = 0
    sent_len = len(sentence)
    phrase_index = 0
    phrase_len = len(phrase)
    while sent_index < sent_len and phrase_index < phrase_len:
        if str(sentence[sent_index]).lower() == str(phrase[phrase_index]).lower():
            sent_index += 1
            phrase_index += 1
            if phrase_index == phrase_len:
                # adjust start to include vocab item(s)
                return sent_index - phrase_len, sent_index  # a tuple
        else:
            sent_index = sent_index - phrase_index + 1
            phrase_index = 0
    return None
