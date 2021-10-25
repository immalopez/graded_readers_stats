##############################################################################
#                                  UTILITIES                                 #
##############################################################################

def vocab_item_to_key(vocab_item: [str]):
    """Gets rid of 'None'-type items, so that vocabulary units containing them
    can still be used as keys in a dictionary."""
    sanitized = filter(lambda word: word is not None, vocab_item)
    return '_'.join(sanitized)
