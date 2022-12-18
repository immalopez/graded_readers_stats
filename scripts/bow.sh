venv/bin/python main.py bag-of-words --strip-named-entities "Data/Graded readers list.csv" | tee output/BOW_graded_readers_ner.log
venv/bin/python main.py bag-of-words --strip-named-entities "Data/Literature list.csv" | tee output/BOW_literature_ner.log
