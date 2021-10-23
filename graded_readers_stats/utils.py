##############################################################################
#                                  UTILITIES                                 #
##############################################################################

def get_vocab_freq_for_level(vocab_item, level, vocab_counts_per_level):
    """Computes the relative frequency of each vocabulary item it is passed."""
    key = vocab_item_to_key(vocab_item[0])
    level_counts = vocab_counts_per_level[key][level]
    return level_counts[0] / level_counts[1]


def vocab_item_to_key(vocab_item: [str]):
    """Gets rid of 'None'-type items, so that vocabulary units containing them
    can still be used as keys in a dictionary."""
    sanitized = filter(lambda word: word is not None, vocab_item)
    return '_'.join(sanitized)
