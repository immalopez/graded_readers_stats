from lexicalrichness import LexicalRichness


def get_lexical_richness(text):
    """
    Calculates the lexical richness of the text.
    """
    lex = LexicalRichness(text)
    return {
        "msttr": lex.msttr(segment_window=100),
        "mattr": lex.mattr(window_size=100),
        "ttr": lex.ttr,
        "rttr": lex.rttr,
        "cttr": lex.cttr,
        "mtld": lex.mtld(threshold=0.72),
        "hdd": lex.hdd(draws=42),
        "Herdan": lex.Herdan,
        "Summer": lex.Summer,
        "Dugast": lex.Dugast,
        "Maas": lex.Maas,
    }
