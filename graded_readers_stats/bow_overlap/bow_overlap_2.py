###############################################################################
# Goal: Check the overlap between BOW output and vocabulary terms.
###############################################################################
import os
from dataclasses import dataclass
from functools import reduce
from os import path

import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

from graded_readers_stats.data import read_pandas_csv


def print_overlap(
        header: str,
        words_vocab: set[str],
        words_bow: set[str]
) -> None:
    print("---")
    print(header)
    words_overlap = words_vocab.intersection(words_bow)
    print("Overlap count: ", len(words_overlap))
    print(sorted(words_overlap))


###############################################################################
# DATA
###############################################################################

vocabulary_csv_path = "../../Data/Vocabulary list.csv"
vocab_df = read_pandas_csv(vocabulary_csv_path)
vocab_df["Lexical item"] = vocab_df["Lexical item"].apply(
    lambda x: x.lower().strip()
)

vocab_all_words = set(vocab_df["Lexical item"])
vocab_level1_words = set(
    vocab_df[vocab_df["Level"] == "A1-A2"]["Lexical item"])
vocab_level2_words = set(vocab_df[vocab_df["Level"] == "B1"]["Lexical item"])
vocab_level3_words = set(vocab_df[vocab_df["Level"] == "B2"]["Lexical item"])

# Graded
bow_graded_1_all = {
    "camping": 0.254441,
    "cuadro": 0.243715,
    "ser": 0.232683,
    "sinagoga": 0.216836,
    "contesto": 0.212997,
    "decir": 0.191669,
    "verónicamente": 0.187560,
    "agencia": 0.176369,
    "lunes": 0.173179,
    "español": 0.172295,
    "ladrón": 0.169576,
    "inspector": 0.161051,
    "museo": 0.154973,
    "gustar": 0.153468,
    "billete": 0.145824,
    "llave": 0.140848,
    "ir": 0.137703,
    "señor": 0.134744,
    "leer": 0.132342,
    "pregunta": 0.122089,
    "tener": 0.121691,
    "vacaciones": 0.120589,
    "volar": 0.118761,
    "artículo": 0.118587,
    "fuente": 0.116344,
    "ventanilla": 0.115968,
    "trabajar": 0.113636,
    "socio": 0.113518,
    "día": 0.109525,
    "aquí": 0.108708,
    "inscripción": 0.108418,
    "ahora": 0.107505,
    "azul": 0.106248,
    "interesante": 0.104992,
    "quién": 0.104782,
    "emplear": 0.104432,
    "escribir": 0.103138,
    "chocolate": 0.102757,
    "mañana": 0.100710,
    "café": 0.099942,
    "caso": 0.098217,
    "robo": 0.094405,
    "oca": 0.090374,
    "hucho": 0.089461,
    "policía": 0.089246,
    "robar": 0.088208,
    "buen": 0.087350,
    "pensar": 0.086521,
    "periódico": 0.086327,
    "semana": 0.085892,
    "bonito": 0.085612,
    "cano": 0.084814,
    "jacinto": 0.084814,
    "comprar": 0.084368,
    "cine": 0.083861,
    "agosto": 0.083122,
    "típico": 0.082856,
    "amigo": 0.082696,
    "comisaría": 0.080110,
    "joya": 0.080091,
    "mujer": -0.158744,
    "si": -0.155742,
    "haber": -0.137623,
    "pueblo": -0.133730,
    "disco": -0.130698,
    "ver": -0.127942,
    "marinero": -0.127750,
    "mar": -0.116219,
    "balsa": -0.112708,
    "aquel": -0.103854,
    "ropa": -0.096054,
    "pez": -0.093640,
    "fuego": -0.092370,
    "aunque": -0.090985,
    "departamento": -0.085962,
    "propio": -0.078994,
    "responder": -0.077854,
    "hombre": -0.077356,
    "dar": -0.075596,
    "batalla": -0.074396,
    "moto": -0.074002,
    "fuerza": -0.072682,
    "empresa": -0.072035,
    "rambla": -0.071948,
    "encontrar": -0.070953,
    "pronto": -0.070697,
    "marido": -0.069451,
    "zapato": -0.067889,
    "tíaú": -0.067590,
    "padre": -0.067585,
    "cuenta": -0.067442,
    "multitud": -0.066238,
    "tiempo": -0.065045,
    "varios": -0.064097,
    "camión": -0.063768,
    "fondo": -0.063581,
    "nombre": -0.062960,
    "querío": -0.062960,
    "forma": -0.062843,
    "music": -0.062601,
    "hall": -0.062601,
    "gratin": -0.062601,
    "indio": -0.060240,
    "viaje": -0.060057,
    "voz": -0.059567,
    "mundo": -0.059515,
    "isla": -0.059251,
    "embargo": -0.059236,
    "cara": -0.058704,
    "hacia": -0.057691,
    "hora": -0.057372,
    "posible": -0.056877,
    "pieza": -0.056634,
    "entonces": -0.056614,
    "añadir": -0.056555,
    "lanzar": -0.056391,
    "patio": -0.054875,
    "italiano": -0.054632,
    "brasa": -0.053430,
    "seguridad": -0.052807,
}
bow_graded_1_pos = {k: v for k, v in bow_graded_1_all.items() if v > 0}
bow_graded_1_neg = {k: v for k, v in bow_graded_1_all.items() if v < 0}
bow_graded_2_all = {
    "marinero": 0.258622,
    "mujer": 0.236389,
    "pez": 0.188478,
    "fuego": 0.173832,
    "marido": 0.163148,
    "batalla": 0.148766,
    "ropa": 0.143382,
    "río": 0.135701,
    "rambla": 0.135640,
    "vecino": 0.135579,
    "multitud": 0.133680,
    "camión": 0.127514,
    "caña": 0.118337,
    "brasa": 0.113742,
    "pueblo": 0.105969,
    "alcalde": 0.102830,
    "san": 0.101544,
    "hora": 0.100362,
    "empezar": 0.097761,
    "quemar": 0.096890,
    "mar": 0.096640,
    "abierto": 0.094999,
    "tienda": 0.094362,
    "brazo": 0.092479,
    "padre": 0.092271,
    "cámara": 0.091449,
    "hacer": 0.091301,
    "hoguera": 0.090993,
    "lleno": 0.089787,
    "rey": 0.088496,
    "hacia": 0.087956,
    "balcón": 0.086698,
    "plaza": 0.085675,
    "pescar": 0.085672,
    "añadir": 0.085372,
    "lanzar": 0.083851,
    "nuevo": 0.082635,
    "horchata": 0.081384,
    "susana": 0.080767,
    "dentro": 0.080548,
    "timbre": 0.079724,
    "allá": 0.079710,
    "proteger": 0.079621,
    "ver": 0.079424,
    "puerta": 0.078832,
    "respondí": 0.077608,
    "piscina": 0.077410,
    "izquierda": 0.076577,
    "recuerdo": 0.076260,
    "joven": 0.076196,
    "camiseta": 0.075916,
    "alrededor": 0.075413,
    "secuestrado": 0.074986,
    "pieza": 0.074127,
    "primo": 0.073218,
    "después": 0.072728,
    "fuerza": 0.072036,
    "depósito": 0.071663,
    "enfermo": 0.071559,
    "cabina": 0.070825,
    "ser": -0.246809,
    "señor": -0.158052,
    "decir": -0.152244,
    "gustar": -0.150980,
    "saber": -0.147707,
    "balsa": -0.132097,
    "contesto": -0.129026,
    "camping": -0.125428,
    "pregunta": -0.125133,
    "cuadro": -0.124889,
    "sinagoga": -0.113905,
    "querer": -0.110606,
    "ladrón": -0.108029,
    "haber": -0.107444,
    "cosa": -0.099239,
    "inspector": -0.097166,
    "verónicamente": -0.093045,
    "interesar": -0.091608,
    "bueno": -0.089545,
    "ahora": -0.087136,
    "poner": -0.084785,
    "español": -0.084171,
    "agencia": -0.083018,
    "entender": -0.082769,
    "trabajar": -0.082185,
    "aquí": -0.082170,
    "lunes": -0.080600,
    "vacaciones": -0.076653,
    "buen": -0.076638,
    "extranjero": -0.074483,
    "artículo": -0.074171,
    "arte": -0.073568,
    "ciudad": -0.072079,
    "llave": -0.070651,
    "pequeño": -0.070201,
    "robo": -0.069985,
    "museo": -0.069563,
    "hoy": -0.068892,
    "billete": -0.068469,
    "claro": -0.068144,
    "hablar": -0.067630,
    "comprar": -0.066897,
    "departamento": -0.066510,
    "music": -0.066273,
    "hall": -0.066273,
    "gratin": -0.066273,
    "semana": -0.065540,
    "arqueólogo": -0.065115,
    "disco": -0.064866,
    "interesante": -0.063105,
    "quién": -0.062731,
    "indio": -0.062477,
    "escribir": -0.062011,
    "querío": -0.061613,
    "sonreír": -0.061081,
    "ventanilla": -0.060920,
    "leer": -0.060397,
    "fuente": -0.059897,
    "difícil": -0.058901,
    "robar": -0.058761,
}
bow_graded_2_pos = {k: v for k, v in bow_graded_2_all.items() if v > 0}
bow_graded_2_neg = {k: v for k, v in bow_graded_2_all.items() if v < 0}
bow_graded_3_all = {
    "haber": 0.245066,
    "balsa": 0.244804,
    "disco": 0.195564,
    "departamento": 0.152472,
    "si": 0.144010,
    "cosa": 0.137019,
    "gratin": 0.128874,
    "hall": 0.128874,
    "music": 0.128874,
    "querío": 0.124573,
    "indio": 0.122717,
    "tíaú": 0.120089,
    "jesuita": 0.108351,
    "religioso": 0.108351,
    "propio": 0.106327,
    "forma": 0.106228,
    "aunque": 0.099274,
    "italiano": 0.096254,
    "pronto": 0.093629,
    "niña": 0.090770,
    "velar": 0.089020,
    "aquel": 0.087582,
    "social": 0.086216,
    "hombre": 0.083622,
    "nativo": 0.083495,
    "nombre": 0.081870,
    "guaraníe": 0.081264,
    "industria": 0.081264,
    "bueno": 0.080144,
    "personal": 0.079989,
    "grupo": 0.079919,
    "expedición": 0.079820,
    "moto": 0.078056,
    "europeo": 0.078055,
    "atreviste": 0.077858,
    "insumiso": 0.077858,
    "belle": 0.077324,
    "dama": 0.077324,
    "mantener": 0.077324,
    "époque": 0.077324,
    "cuyo": 0.075524,
    "arte": 0.074587,
    "sistema": 0.074150,
    "patio": 0.071713,
    "saber": 0.070530,
    "música": 0.070075,
    "americano": 0.070002,
    "artístico": 0.069959,
    "viernes": 0.069799,
    "cambiar": 0.068766,
    "parecer": 0.068470,
    "cuenta": 0.068363,
    "indígena": 0.067996,
    "tratar": 0.067558,
    "culpable": 0.067454,
    "embarcación": 0.066765,
    "polinesio": 0.066765,
    "tronco": 0.066765,
    "hablar": 0.066343,
    "biblioteca": 0.065658,
    "marinero": -0.130872,
    "camping": -0.129013,
    "cuadro": -0.118825,
    "sinagoga": -0.102932,
    "bar": -0.099245,
    "dentro": -0.096669,
    "pez": -0.094838,
    "verónicamente": -0.094515,
    "ventana": -0.094210,
    "marido": -0.093697,
    "agencia": -0.093351,
    "ir": -0.092925,
    "lunes": -0.092579,
    "sol": -0.092202,
    "fiesta": -0.090373,
    "plaza": -0.090272,
    "chico": -0.089452,
    "español": -0.088124,
    "joya": -0.087963,
    "cerca": -0.087783,
    "calor": -0.086210,
    "río": -0.086082,
    "rey": -0.085686,
    "museo": -0.085410,
    "vecino": -0.085300,
    "contesto": -0.083971,
    "alcalde": -0.083517,
    "coger": -0.083456,
    "fuego": -0.081462,
    "aire": -0.080756,
    "nuevo": -0.079908,
    "mujer": -0.077645,
    "billete": -0.077355,
    "batalla": -0.074370,
    "sentar": -0.072621,
    "leer": -0.071946,
    "caso": -0.071258,
    "ojo": -0.070897,
    "allí": -0.070563,
    "abierto": -0.070339,
    "llave": -0.070197,
    "caña": -0.070193,
    "volar": -0.070055,
    "nervioso": -0.069465,
    "entrar": -0.068849,
    "empezar": -0.068833,
    "verde": -0.068465,
    "doce": -0.068321,
    "dos": -0.068294,
    "multitud": -0.067442,
    "día": -0.066904,
    "tres": -0.066618,
    "tener": -0.066298,
    "mañana": -0.066271,
    "pensar": -0.064095,
    "inspector": -0.063885,
    "camión": -0.063745,
    "alto": -0.063707,
    "rambla": -0.063691,
    "nadie": -0.063299,
}
bow_graded_3_pos = {k: v for k, v in bow_graded_3_all.items() if v > 0}
bow_graded_3_neg = {k: v for k, v in bow_graded_3_all.items() if v < 0}
bow_graded_all = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_all, bow_graded_2_all, bow_graded_3_all)
)
bow_graded_pos = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_pos, bow_graded_2_pos, bow_graded_3_pos)
)
bow_graded_neg = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_neg, bow_graded_2_neg, bow_graded_3_neg)
)

# Literature
bow_litera_1_all = {
    "bruja": 0.184563,
    "príncipe": 0.180898,
    "día": 0.170378,
    "bosque": 0.161877,
    "fábrica": 0.144549,
    "niña": 0.144309,
    "lámpara": 0.129290,
    "gato": 0.114915,
    "balleno": 0.108313,
    "barco": 0.106248,
    "ricito": 0.105651,
    "aprender": 0.101966,
    "patito": 0.098367,
    "poder": 0.095586,
    "lugar": 0.093032,
    "vivir": 0.091337,
    "así": 0.089844,
    "nuevo": 0.089683,
    "casa": 0.085561,
    "invento": 0.085142,
    "oveja": 0.085131,
    "encontrar": 0.084539,
    "volar": 0.082632,
    "cuento": 0.082575,
    "cada": 0.081342,
    "tres": 0.080173,
    "carabás": 0.078734,
    "marqués": 0.078734,
    "pan": 0.078164,
    "hada": 0.077100,
    "comenzar": 0.074616,
    "contar": 0.073317,
    "cole": 0.071542,
    "coronavir": 0.071542,
    "entrenar": 0.070203,
    "animal": 0.069627,
    "convertir": 0.068907,
    "reina": 0.068726,
    "mayor": 0.068658,
    "pato": 0.068297,
    "princesa": 0.068016,
    "amar": 0.066909,
    "espantapájaro": 0.065921,
    "buscar": 0.064533,
    "osa": 0.064049,
    "polar": 0.064049,
    "pasaron": 0.063312,
    "chocolate": 0.063122,
    "ayudar": 0.063001,
    "madrastra": 0.062737,
    "lejano": 0.062651,
    "madera": 0.062426,
    "rey": 0.062411,
    "planeta": 0.062404,
    "allí": 0.061775,
    "genio": 0.061024,
    "dinosaurios": 0.060634,
    "faro": 0.060634,
    "cumplir": 0.060338,
    "hogar": 0.059047,
    "ser": -0.576968,
    "decir": -0.511819,
    "haber": -0.487120,
    "muñeco": -0.280220,
    "rosa": -0.265344,
    "cohete": -0.237241,
    "molinero": -0.234156,
    "gigante": -0.227198,
    "tener": -0.219707,
    "rojo": -0.209337,
    "preguntar": -0.190686,
    "ruiseñor": -0.189648,
    "jardín": -0.188538,
    "pelo": -0.158143,
    "pues": -0.158084,
    "mano": -0.151448,
    "señor": -0.141160,
    "gritar": -0.134449,
    "responder": -0.129860,
    "mismo": -0.118760,
    "cabeza": -0.116278,
    "habitación": -0.116083,
    "ojo": -0.115210,
    "carretilla": -0.114835,
    "estudiante": -0.112386,
    "creer": -0.111974,
    "hacia": -0.111022,
    "rosal": -0.110723,
    "árbol": -0.110349,
    "replicar": -0.108668,
    "exclamar": -0.108192,
    "sristino": -0.103394,
    "dos": -0.100224,
    "brazo": -0.096919,
    "blanco": -0.095348,
    "embargo": -0.092170,
    "hablar": -0.089409,
    "alguno": -0.089369,
    "gran": -0.088395,
    "vez": -0.087604,
    "entonces": -0.087238,
    "niño": -0.087189,
    "estatua": -0.086321,
    "parecer": -0.085807,
    "madre": -0.084912,
    "miré": -0.082392,
    "rubí": -0.081426,
    "pensar": -0.080273,
    "poner": -0.080177,
    "ir": -0.077843,
    "luego": -0.076736,
    "después": -0.076651,
    "realmente": -0.075605,
    "levantar": -0.074727,
    "hombro": -0.073395,
    "saber": -0.073160,
    "verdaderamente": -0.072975,
    "carta": -0.072767,
    "pardillo": -0.072527,
    "cara": -0.071749,
}
bow_litera_1_pos = {k: v for k, v in bow_litera_1_all.items() if v > 0}
bow_litera_1_neg = {k: v for k, v in bow_litera_1_all.items() if v < 0}
bow_litera_2_all = {
    "decir": 0.364433,
    "muñeco": 0.344742,
    "rosa": 0.308141,
    "molinero": 0.286683,
    "cohete": 0.284379,
    "gigante": 0.268585,
    "rojo": 0.244555,
    "jardín": 0.226611,
    "ruiseñor": 0.216656,
    "ser": 0.199872,
    "pelo": 0.191289,
    "señor": 0.190739,
    "mano": 0.162709,
    "niño": 0.156712,
    "gritar": 0.152894,
    "cabeza": 0.138693,
    "carretilla": 0.135086,
    "preguntar": 0.134120,
    "rosal": 0.126739,
    "tener": 0.126468,
    "sristino": 0.125715,
    "árbol": 0.115917,
    "estudiante": 0.113019,
    "exclamar": 0.112334,
    "blanco": 0.110751,
    "pues": 0.109348,
    "habitación": 0.109298,
    "replicar": 0.109020,
    "padre": 0.107195,
    "hacia": 0.104999,
    "estatua": 0.100695,
    "gran": 0.099804,
    "ojo": 0.097294,
    "rubí": 0.096370,
    "coger": 0.090646,
    "realmente": 0.087742,
    "entonces": 0.087628,
    "espino": 0.086146,
    "pardillo": 0.085317,
    "madre": 0.083727,
    "hombro": 0.082858,
    "bien": 0.082245,
    "brazo": 0.079222,
    "pequeño": 0.079097,
    "primavera": 0.079016,
    "cara": 0.078040,
    "corazón": 0.073129,
    "negro": 0.072539,
    "capullo": 0.072126,
    "miré": 0.071769,
    "nieve": 0.071675,
    "rata": 0.071176,
    "candela": 0.071095,
    "creer": 0.069750,
    "perro": 0.068725,
    "río": 0.068505,
    "caer": 0.068477,
    "voló": 0.068131,
    "responder": 0.067964,
    "poner": 0.067437,
    "día": -0.158094,
    "poder": -0.150535,
    "bruja": -0.141047,
    "príncipe": -0.117841,
    "bosque": -0.117744,
    "fábrica": -0.111695,
    "casa": -0.108401,
    "encontrar": -0.103378,
    "niña": -0.099656,
    "lámpara": -0.098817,
    "lugar": -0.094721,
    "gato": -0.090876,
    "hombre": -0.085072,
    "así": -0.084912,
    "barco": -0.084196,
    "balleno": -0.082812,
    "ricito": -0.082295,
    "nuevo": -0.081604,
    "mujer": -0.079192,
    "vivir": -0.078675,
    "comenzar": -0.077001,
    "aprender": -0.076904,
    "cada": -0.076287,
    "patito": -0.071231,
    "aquel": -0.070121,
    "mayor": -0.069681,
    "tratar": -0.067992,
    "invento": -0.065981,
    "tres": -0.064929,
    "oveja": -0.064628,
    "cuento": -0.062125,
    "momento": -0.061878,
    "marqués": -0.061805,
    "carabás": -0.061805,
    "pan": -0.061238,
    "hada": -0.060070,
    "buscar": -0.059478,
    "taller": -0.057187,
    "volar": -0.057166,
    "contar": -0.056822,
    "animal": -0.054833,
    "salir": -0.054786,
    "amar": -0.054433,
    "coronavir": -0.054204,
    "cole": -0.054204,
    "reina": -0.054104,
    "madrastra": -0.054043,
    "entrenar": -0.053954,
    "sueño": -0.053945,
    "hora": -0.053083,
    "mar": -0.052914,
    "pasaron": -0.052396,
    "convertir": -0.052145,
    "comer": -0.052084,
    "deseo": -0.051700,
    "decidir": -0.051327,
    "lejano": -0.051101,
    "carta": -0.050756,
    "allí": -0.050074,
    "polar": -0.049862,
}
bow_litera_2_pos = {k: v for k, v in bow_litera_2_all.items() if v > 0}
bow_litera_2_neg = {k: v for k, v in bow_litera_2_all.items() if v < 0}
bow_litera_3_all = {
    "haber": 0.454293,
    "ser": 0.377096,
    "decir": 0.147385,
    "carta": 0.123523,
    "mismo": 0.103313,
    "cierto": 0.098803,
    "palabra": 0.094581,
    "hacer": 0.093854,
    "tener": 0.093240,
    "si": 0.088016,
    "vez": 0.083913,
    "pensar": 0.083622,
    "saber": 0.081903,
    "después": 0.079752,
    "aquel": 0.078068,
    "embargo": 0.077791,
    "modo": 0.077613,
    "parecer": 0.076566,
    "momento": 0.074959,
    "alguno": 0.073596,
    "tiempo": 0.072926,
    "instante": 0.071774,
    "tomar": 0.071620,
    "recuerdo": 0.071149,
    "estancia": 0.069287,
    "fué": 0.069157,
    "idea": 0.067277,
    "hablar": 0.066250,
    "hombre": 0.062397,
    "tal": 0.062274,
    "responder": 0.061896,
    "pregunta": 0.060368,
    "deber": 0.057038,
    "escena": 0.056695,
    "preguntar": 0.056566,
    "querer": 0.055571,
    "poder": 0.054948,
    "mujer": 0.054628,
    "quizá": 0.054387,
    "rublo": 0.052755,
    "cómo": 0.051923,
    "persona": 0.051561,
    "posibilidad": 0.050743,
    "ahora": 0.050310,
    "advertir": 0.050299,
    "sentir": 0.050158,
    "semejante": 0.049136,
    "mirada": 0.048967,
    "pues": 0.048736,
    "sentimiento": 0.047661,
    "duda": 0.047076,
    "conversación": 0.047038,
    "objeto": 0.046796,
    "rostro": 0.046658,
    "seguir": 0.045873,
    "frase": 0.045415,
    "venir": 0.045102,
    "posible": 0.044487,
    "parte": 0.043798,
    "aunque": 0.043578,
    "niño": -0.069523,
    "muñeco": -0.064522,
    "príncipe": -0.063056,
    "molinero": -0.052527,
    "señor": -0.049578,
    "padre": -0.048433,
    "cohete": -0.047138,
    "niña": -0.044653,
    "bosque": -0.044133,
    "bruja": -0.043516,
    "rosa": -0.042797,
    "gigante": -0.041387,
    "pequeño": -0.040966,
    "jardín": -0.038073,
    "rojo": -0.035218,
    "pelo": -0.033146,
    "fábrica": -0.032854,
    "rey": -0.032357,
    "madera": -0.032330,
    "feliz": -0.031932,
    "amarillo": -0.030553,
    "lámpara": -0.030473,
    "coger": -0.028758,
    "agua": -0.028416,
    "patito": -0.027136,
    "princesa": -0.027096,
    "ruiseñor": -0.027008,
    "joven": -0.026961,
    "oro": -0.026818,
    "balleno": -0.025501,
    "volar": -0.025466,
    "compartir": -0.025326,
    "bello": -0.025256,
    "aprender": -0.025063,
    "gato": -0.024039,
    "junto": -0.023980,
    "voló": -0.023551,
    "ricito": -0.023357,
    "cabeza": -0.022415,
    "sristino": -0.022321,
    "barco": -0.022052,
    "jugar": -0.020576,
    "oveja": -0.020503,
    "cuento": -0.020450,
    "armario": -0.020355,
    "carretilla": -0.020251,
    "pato": -0.020014,
    "flor": -0.019665,
    "nieve": -0.019554,
    "ayudar": -0.019457,
    "primavera": -0.019420,
    "invento": -0.019161,
    "espino": -0.018567,
    "gustar": -0.018449,
    "gritar": -0.018446,
    "invierno": -0.018387,
    "amiga": -0.018357,
    "saco": -0.018023,
    "llevar": -0.017967,
    "coronavir": -0.017338,
}
bow_litera_3_pos = {k: v for k, v in bow_litera_3_all.items() if v > 0}
bow_litera_3_neg = {k: v for k, v in bow_litera_3_all.items() if v < 0}
bow_litera_all = reduce(
    lambda x, y: dict(x, **y),
    (bow_litera_1_all, bow_litera_2_all, bow_litera_3_all)
)
bow_litera_pos = reduce(
    lambda x, y: dict(x, **y),
    (bow_litera_1_pos, bow_litera_2_pos, bow_litera_3_pos)
)
bow_litera_neg = reduce(
    lambda x, y: dict(x, **y),
    (bow_litera_1_neg, bow_litera_2_neg, bow_litera_3_neg)
)

# Graded + Literature
bow_all = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_all, bow_graded_2_all, bow_graded_3_all,
     bow_litera_1_all, bow_litera_2_all, bow_litera_3_all)
)
bow_pos = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_pos, bow_graded_2_pos, bow_graded_3_pos,
     bow_litera_1_pos, bow_litera_2_pos, bow_litera_3_pos)
)
bow_neg = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_neg, bow_graded_2_neg, bow_graded_3_neg,
     bow_litera_1_neg, bow_litera_2_neg, bow_litera_3_neg)
)


##############################################################################
# WordCloud generation
##############################################################################


@dataclass
class WordCloudSection:
    mask: str
    output: str
    wordToFreqMap: dict[str: float]


def draw_wc_section(section: WordCloudSection):
    d = os.getcwd()
    mask = np.array(Image.open(path.join(d, section.mask)))
    wc = WordCloud(
        mode="RGBA",
        background_color=None,
        width=2800,
        height=2800,
        mask=mask
    )
    if len(section.wordToFreqMap.items()):
        wc.generate_from_frequencies(section.wordToFreqMap)
        wc.to_file(path.join(d, section.output))


freqs_pos = {
    "to1": {
        k: v
        for k, v in bow_graded_1_pos.items()
        if k in set(bow_graded_1_pos.keys()).difference(vocab_level1_words)
    },
    "to2": {
        k: v
        for k, v in bow_graded_2_pos.items()
        if k in set(bow_graded_2_pos.keys()).difference(vocab_level2_words)
    },
    "to3": {
        k: v
        for k, v in bow_graded_3_pos.items()
        if k in set(bow_graded_3_pos.keys()).difference(vocab_level3_words)
    },
    "ti1": {
        k: v
        for k, v in bow_graded_1_pos.items()
        if k in set(bow_graded_1_pos.keys()).intersection(vocab_level1_words)
    },
    "ti2": {
        k: v
        for k, v in bow_graded_2_pos.items()
        if k in set(bow_graded_2_pos.keys()).intersection(vocab_level2_words)
    },
    "ti3": {
        k: v
        for k, v in bow_graded_3_pos.items()
        if k in set(bow_graded_3_pos.keys()).intersection(vocab_level3_words)
    },
    "bo1": {
        k: v
        for k, v in bow_litera_1_pos.items()
        if k in set(bow_litera_1_pos.keys()).difference(vocab_level1_words)
    },
    "bo2": {
        k: v
        for k, v in bow_litera_2_pos.items()
        if k in set(bow_litera_2_pos.keys()).difference(vocab_level2_words)
    },
    "bo3": {
        k: v
        for k, v in bow_litera_3_pos.items()
        if k in set(bow_litera_3_pos.keys()).difference(vocab_level3_words)
    },
    "bi1": {
        k: v
        for k, v in bow_litera_1_pos.items()
        if k in set(bow_litera_1_pos.keys()).intersection(vocab_level1_words)
    },
    "bi2": {
        k: v
        for k, v in bow_litera_2_pos.items()
        if k in set(bow_litera_2_pos.keys()).intersection(vocab_level2_words)
    },
    "bi3": {
        k: v
        for k, v in bow_litera_3_pos.items()
        if k in set(bow_litera_3_pos.keys()).intersection(vocab_level3_words)
    },
}
freqs_neg = {
    "to1": {
        k: v
        for k, v in bow_graded_1_neg.items()
        if k in set(bow_graded_1_neg.keys()).difference(vocab_level1_words)
    },
    "to2": {
        k: v
        for k, v in bow_graded_2_neg.items()
        if k in set(bow_graded_2_neg.keys()).difference(vocab_level2_words)
    },
    "to3": {
        k: v
        for k, v in bow_graded_3_neg.items()
        if k in set(bow_graded_3_neg.keys()).difference(vocab_level3_words)
    },
    "ti1": {
        k: v
        for k, v in bow_graded_1_neg.items()
        if k in set(bow_graded_1_neg.keys()).intersection(vocab_level1_words)
    },
    "ti2": {
        k: v
        for k, v in bow_graded_2_neg.items()
        if k in set(bow_graded_2_neg.keys()).intersection(vocab_level2_words)
    },
    "ti3": {
        k: v
        for k, v in bow_graded_3_neg.items()
        if k in set(bow_graded_3_neg.keys()).intersection(vocab_level3_words)
    },
    "bo1": {
        k: v
        for k, v in bow_litera_1_neg.items()
        if k in set(bow_litera_1_neg.keys()).difference(vocab_level1_words)
    },
    "bo2": {
        k: v
        for k, v in bow_litera_2_neg.items()
        if k in set(bow_litera_2_neg.keys()).difference(vocab_level2_words)
    },
    "bo3": {
        k: v
        for k, v in bow_litera_3_neg.items()
        if k in set(bow_litera_3_neg.keys()).difference(vocab_level3_words)
    },
    "bi1": {
        k: v
        for k, v in bow_litera_1_neg.items()
        if k in set(bow_litera_1_neg.keys()).intersection(vocab_level1_words)
    },
    "bi2": {
        k: v
        for k, v in bow_litera_2_neg.items()
        if k in set(bow_litera_2_neg.keys()).intersection(vocab_level2_words)
    },
    "bi3": {
        k: v
        for k, v in bow_litera_3_neg.items()
        if k in set(bow_litera_3_neg.keys()).intersection(vocab_level3_words)
    },
}

for name, freqs in [("pos", freqs_pos), ("neg", freqs_neg)]:
    for index in [1, 2, 3]:
        for section in ["to", "ti", "bo", "bi"]:
            sectionName = f"{section}{index}"
            draw_wc_section(
                WordCloudSection(
                    mask=f"masks/{sectionName}.jpg",
                    output=f"output/{name}/{sectionName}.png",
                    wordToFreqMap=freqs[sectionName]
                )
            )


print("Done!")
