venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Graded readers list.csv" Inicial 2>&1 | tee output/Inicial.log &
venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Graded readers list.csv" Intermedio 2>&1 | tee output/Intermedio.log &
venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Graded readers list.csv" Avanzado 2>&1 | tee output/Avanzado.log &
venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Literature list.csv" Infantil 2>&1 | tee output/Infantil.log &
venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Literature list.csv" Juvenil 2>&1 | tee output/Juvenil.log &
venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Literature list.csv" Adulta 2>&1 | tee output/Adulta.log &
venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Native list.csv" Native 2>&1 | tee output/Native.log &
wait

venv/bin/python main.py merge-output
