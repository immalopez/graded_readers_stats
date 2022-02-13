venv/bin/python main.py analyze "Data_trial/Vocabulary list.csv" "Data_trial/Graded readers list.csv" Inicial | tee output/Inicial.log &
venv/bin/python main.py analyze "Data_trial/Vocabulary list.csv" "Data_trial/Graded readers list.csv" Intermedio | tee output/Intermedio.log &
venv/bin/python main.py analyze "Data_trial/Vocabulary list.csv" "Data_trial/Graded readers list.csv" Avanzado | tee output/Avanzado.log &
venv/bin/python main.py analyze "Data_trial/Vocabulary list.csv" "Data_trial/Literature list.csv" Infantil | tee output/Infantil.log &
venv/bin/python main.py analyze "Data_trial/Vocabulary list.csv" "Data_trial/Literature list.csv" Juvenil | tee output/Juvenil.log &
venv/bin/python main.py analyze "Data_trial/Vocabulary list.csv" "Data_trial/Literature list.csv" Adulta | tee output/Adulta.log &
venv/bin/python main.py analyze "Data_trial/Vocabulary list.csv" "Data_trial/Native list.csv" Native | tee output/Native.log &
wait

venv/bin/python main.py merge-output
