venv/bin/python main.py analyze "data/vocabulary.csv" "data/readers/readers.csv" Inicial | tee output/readers_1.log &
venv/bin/python main.py analyze "data/vocabulary.csv" "data/readers/readers.csv" Intermedio | tee output/readers_2.log &
venv/bin/python main.py analyze "data/vocabulary.csv" "data/readers/readers.csv" Avanzado | tee output/readers_3.log &
venv/bin/python main.py analyze "data/vocabulary.csv" "data/literature/literature.csv" Infantil | tee output/literature_1.log &
venv/bin/python main.py analyze "data/vocabulary.csv" "data/literature/literature.csv" Juvenil | tee output/literature_2.log &
venv/bin/python main.py analyze "data/vocabulary.csv" "data/literature/literature.csv" Adulta | tee output/literature_3.log &
venv/bin/python main.py analyze "data/vocabulary.csv" "data/native/native.csv" Native | tee output/native.log &
venv/bin/python main.py terms_by_group "data/vocabulary.csv" --corpus_path="data/readers/readers.csv" --corpus_path="data/literature/literature.csv" --corpus_path="data/native/native.csv" | tee output/terms_by_group.log &
wait

venv/bin/python main.py merge-output
venv/bin/python main.py bag-of-words "data/readers/readers.csv" | tee output/readers_bow.log
venv/bin/python main.py bag-of-words "data/literature/literature.csv" | tee output/literature_bow.log
