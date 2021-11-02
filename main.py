###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################

from graded_readers_stats import data, frequency, preprocess

# Load data
vocabulary, readers = data.load(trial=True)

# Preprocess data
readers = preprocess.run(readers, preprocess.text_analysis_pipeline)
vocabulary = preprocess.run(vocabulary, preprocess.vocabulary_pipeline)

# Vocabulary Frequencies
readers_by_level = readers.groupby(preprocess.LEVEL)
frequency.count_phrases_in_sentences_by_groups(
    vocabulary, readers_by_level, column_prefix='Reader_'
)

# Vocabulary's Context Frequencies
frequency.collect_context_for_phrases_in_texts(
    vocabulary, readers, column_prefix='Reader_'
)
frequency.count_context_in_sentences_by_groups(
    vocabulary, readers_by_level, column_prefix='Reader_Context_'
)



# from stanza.server import CoreNLPClient
#
# # set up the client
# with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse'], timeout=30000, memory='16G') as client:
#     # submit the request to the server
#     ann = client.annotate("A Plamen le gustan las motos.")
#
#     # get the first sentence
#     sentence = ann.sentence[0]
#
#     # get the constituency parse of the first sentence
#     print('---')
#     print('constituency parse of first sentence')
#     constituency_parse = sentence.parseTree
#     print(constituency_parse)
#
#     # get the first subtree of the constituency parse
#     print('---')
#     print('first subtree of constituency parse')
#     print(constituency_parse.child[0])


