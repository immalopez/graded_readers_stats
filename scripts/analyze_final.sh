venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Graded readers list.csv" Inicial | tee output/Inicial.log &
venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Graded readers list.csv" Intermedio | tee output/Intermedio.log &
venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Graded readers list.csv" Avanzado | tee output/Avanzado.log &
venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Literature list.csv" Infantil | tee output/Infantil.log &
venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Literature list.csv" Juvenil | tee output/Juvenil.log &
venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Literature list.csv" Adulta | tee output/Adulta.log &
venv/bin/python main.py analyze "Data/Vocabulary list.csv" "Data/Native list.csv" Native | tee output/Native.log &
wait
