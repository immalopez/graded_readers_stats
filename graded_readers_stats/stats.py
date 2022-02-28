from lexicalrichness import LexicalRichness


def get_msttr(text):
    """
    Calculates the lexical richness of the text.
    """
    lexical_richness = LexicalRichness(text)
    return {'msttr': lexical_richness.msttr(segment_window=3)}
