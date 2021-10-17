

def get_vocab_freq_for_level(vocab_item, level, vocab_counts_per_level):
    key = vocab_item_to_key(vocab_item[0])
    level_counts = vocab_counts_per_level[key][level]
    return level_counts[0] / level_counts[1]


def vocab_item_to_key(vocab_item: [str]):
    sanitized = filter(lambda word: word is not None, vocab_item)
    return '_'.join(sanitized)


print(vocab_item_to_key([None, 'hello', None, 'world', None, '!', None]))
