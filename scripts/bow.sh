venv/bin/python main.py bag-of-words --strip-named-entities "Data/Graded readers list.csv" Inicial | tee output/BOW_graded_inicial.log
venv/bin/python main.py bag-of-words --strip-named-entities "Data/Graded readers list.csv" Intermedio | tee output/BOW_graded_intermedio.log
venv/bin/python main.py bag-of-words --strip-named-entities "Data/Graded readers list.csv" Avanzado | tee output/BOW_graded_avanzado.log
venv/bin/python main.py bag-of-words --strip-named-entities "Data/Literature list.csv" Infantil | tee output/BOW_litera_infantil.log
venv/bin/python main.py bag-of-words --strip-named-entities "Data/Literature list.csv" Juvenil | tee output/BOW_litera_juvenil.log
venv/bin/python main.py bag-of-words --strip-named-entities "Data/Literature list.csv" Adulta | tee output/BOW_litera_adulta.log
