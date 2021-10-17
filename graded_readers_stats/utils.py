

def get_vocab_freq_for_level(vocab_item, level, vocab_counts_per_level):
    key = vocab_item_to_key(vocab_item[0])
    level_counts = vocab_counts_per_level[key][level]
    return level_counts[0] / level_counts[1]


def vocab_item_to_key(vocab_item):
    return '_'.join(vocab_item)


