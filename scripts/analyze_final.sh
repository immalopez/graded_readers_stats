venv/bin/python main.py analyze "data/vocabulary.csv" "data/readers/readers.csv" Inicial | tee output/readers_1.log &
venv/bin/python main.py analyze "data/vocabulary.csv" "data/readers/readers.csv" Intermedio | tee output/readers_2.log &
venv/bin/python main.py analyze "data/vocabulary.csv" "data/readers/readers.csv" Avanzado | tee output/readers_3.log &
venv/bin/python main.py analyze "data/vocabulary.csv" "data/literature/literature.csv" Infantil | tee output/literature_1.log &
venv/bin/python main.py analyze "data/vocabulary.csv" "data/literature/literature.csv" Juvenil | tee output/literature_2.log &
venv/bin/python main.py analyze "data/vocabulary.csv" "data/literature/literature.csv" Adulta | tee output/literature_3.log &
venv/bin/python main.py analyze "data/vocabulary.csv" "data/native/native.csv" Native | tee output/native.log &
wait
