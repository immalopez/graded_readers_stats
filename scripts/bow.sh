venv/bin/python main.py bag-of-words "Data/Graded readers list.csv" | tee output/BOW_graded_readers.log
venv/bin/python main.py bag-of-words "Data/Literature list.csv" | tee output/BOW_literature.log
