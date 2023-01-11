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

vocabulary_csv_path = "../../data/vocabulary.csv"
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
    "cuadro": 0.324657,
    "ir": 0.302338,
    "ser": 0.276551,
    "camping": 0.270908,
    "español": 0.248750,
    "sinagoga": 0.242307,
    "museo": 0.234208,
    "agencia": 0.230549,
    "tener": 0.230175,
    "socio": 0.214989,
    "contesto": 0.211373,
    "verónicamente": 0.202773,
    "gustar": 0.195119,
    "pregunta": 0.191982,
    "señor": 0.185731,
    "bolsa": 0.184895,
    "lunes": 0.176062,
    "inspector": 0.172406,
    "ladrón": 0.165604,
    "oficina": 0.162771,
    "billete": 0.155954,
    "ahora": 0.155652,
    "despacho": 0.155425,
    "trabajar": 0.152227,
    "vacaciones": 0.151501,
    "detective": 0.147436,
    "policía": 0.146312,
    "calle": 0.143149,
    "actriz": 0.142494,
    "interesante": 0.142227,
    "caso": 0.141779,
    "serie": 0.139253,
    "señora": 0.138170,
    "llave": 0.135652,
    "playa": 0.135489,
    "edificio": 0.131732,
    "ventanilla": 0.129963,
    "decir": 0.128721,
    "leer": 0.126384,
    "mañana": 0.126328,
    "periódico": 0.122188,
    "inscripción": 0.121153,
    "lago": 0.120898,
    "expresión": 0.119044,
    "brava": 0.118779,
    "pensar": 0.117503,
    "aquí": 0.117158,
    "volar": 0.115788,
    "cliente": 0.115074,
    "artículo": 0.113697,
    "café": 0.111651,
    "azul": 0.110020,
    "extranjero": 0.109905,
    "día": 0.107291,
    "comprar": 0.106982,
    "escalera": 0.106848,
    "bonito": 0.106599,
    "agosto": 0.106277,
    "típico": 0.106136,
    "querer": 0.104599,
    "si": -0.228667,
    "ver": -0.186082,
    "padre": -0.164646,
    "aquel": -0.161435,
    "responder": -0.146329,
    "amar": -0.143637,
    "mosén": -0.143380,
    "merced": -0.143171,
    "mujer": -0.140239,
    "pueblo": -0.135931,
    "luis": -0.134901,
    "madre": -0.133647,
    "morir": -0.131133,
    "mary": -0.130894,
    "isla": -0.125931,
    "marinero": -0.125041,
    "aunque": -0.124205,
    "disco": -0.124035,
    "balsa": -0.112785,
    "dar": -0.109033,
    "camino": -0.101743,
    "voz": -0.099261,
    "hacer": -0.096928,
    "pez": -0.093887,
    "comenzar": -0.088602,
    "departamento": -0.087353,
    "mundo": -0.087314,
    "gatito": -0.081562,
    "fuerza": -0.081506,
    "sino": -0.078404,
    "fondo": -0.076223,
    "batalla": -0.075185,
    "ciego": -0.075162,
    "alguno": -0.074564,
    "ropa": -0.073772,
    "allá": -0.073766,
    "arca": -0.072737,
    "tíaú": -0.072190,
    "gran": -0.071563,
    "niña": -0.070857,
    "hijo": -0.070477,
    "nombre": -0.070243,
    "cosa": -0.069674,
    "lanzar": -0.069548,
    "caballo": -0.069247,
    "fuego": -0.068220,
    "multitud": -0.067566,
    "brazo": -0.066622,
    "vos": -0.066344,
    "lleno": -0.066252,
    "posible": -0.066221,
    "italiano": -0.065957,
    "moto": -0.065613,
    "hombre": -0.065438,
    "niño": -0.065212,
    "cada": -0.064132,
    "miedo": -0.064114,
    "guardia": -0.063873,
    "embargo": -0.063816,
    "alameda": -0.063534,
}
bow_graded_1_pos = {k: v for k, v in bow_graded_1_all.items() if v > 0}
bow_graded_1_neg = {k: v for k, v in bow_graded_1_all.items() if v < 0}
bow_graded_2_all = {
    "mosén": 0.287947,
    "luis": 0.259387,
    "mary": 0.256384,
    "marinero": 0.243205,
    "pez": 0.180463,
    "isla": 0.179027,
    "mujer": 0.176334,
    "ciego": 0.154100,
    "padre": 0.153509,
    "arca": 0.149129,
    "niño": 0.142772,
    "gatito": 0.140615,
    "multitud": 0.133026,
    "alameda": 0.128301,
    "fuego": 0.124941,
    "río": 0.124130,
    "allá": 0.121248,
    "plaza": 0.119683,
    "ropa": 0.116096,
    "aquel": 0.115128,
    "vos": 0.114509,
    "lleno": 0.111355,
    "camión": 0.111194,
    "dar": 0.111147,
    "brasa": 0.110895,
    "empezar": 0.109478,
    "vez": 0.109326,
    "iguano": 0.105578,
    "vecino": 0.104811,
    "brazo": 0.101923,
    "pueblo": 0.101801,
    "amar": 0.100956,
    "cabina": 0.098917,
    "saco": 0.098238,
    "alcalde": 0.098082,
    "hora": 0.097064,
    "rambla": 0.095224,
    "marido": 0.095210,
    "batalla": 0.094018,
    "proteger": 0.093983,
    "meter": 0.093672,
    "abuelo": 0.093611,
    "hacer": 0.093327,
    "monaguillo": 0.092360,
    "aún": 0.091516,
    "ver": 0.091471,
    "entonces": 0.091156,
    "caminar": 0.089649,
    "abierto": 0.089493,
    "enfermo": 0.088905,
    "caña": 0.088438,
    "balcón": 0.087725,
    "cámara": 0.086978,
    "higgins": 0.086382,
    "cuarto": 0.086266,
    "volví": 0.086120,
    "ojo": 0.085490,
    "hoguera": 0.084780,
    "pan": 0.084738,
    "san": 0.084396,
    "decir": -0.314595,
    "ser": -0.298326,
    "señor": -0.215115,
    "querer": -0.187987,
    "cuadro": -0.186589,
    "pregunta": -0.176728,
    "balsa": -0.171174,
    "gustar": -0.166553,
    "camping": -0.152547,
    "sinagoga": -0.148141,
    "detective": -0.145937,
    "contesto": -0.144204,
    "tener": -0.137739,
    "haber": -0.133006,
    "museo": -0.126270,
    "agencia": -0.124710,
    "ir": -0.122028,
    "inspector": -0.119801,
    "socio": -0.116926,
    "verónicamente": -0.114479,
    "español": -0.113613,
    "bueno": -0.112956,
    "trabajar": -0.111540,
    "ladrón": -0.110459,
    "merced": -0.103767,
    "oficina": -0.102981,
    "saber": -0.102513,
    "buen": -0.099637,
    "extranjero": -0.097871,
    "entender": -0.097764,
    "ahora": -0.097445,
    "señora": -0.096024,
    "policía": -0.095861,
    "disco": -0.094479,
    "lunes": -0.094150,
    "arte": -0.092590,
    "aquí": -0.090762,
    "poner": -0.090324,
    "interesante": -0.089873,
    "vacaciones": -0.088907,
    "hablar": -0.088753,
    "despacho": -0.086881,
    "arqueólogo": -0.085425,
    "robo": -0.084836,
    "music": -0.084138,
    "hall": -0.084138,
    "gratin": -0.084138,
    "indio": -0.083731,
    "hotel": -0.083064,
    "bolsa": -0.083037,
    "caso": -0.080993,
    "quién": -0.079588,
    "actriz": -0.079216,
    "billete": -0.078450,
    "historia": -0.078261,
    "interesar": -0.077676,
    "serie": -0.077119,
    "querío": -0.076338,
    "comprar": -0.075915,
    "leer": -0.075528,
}
bow_graded_2_pos = {k: v for k, v in bow_graded_2_all.items() if v > 0}
bow_graded_2_neg = {k: v for k, v in bow_graded_2_all.items() if v < 0}
bow_graded_3_all = {
    "balsa": 0.283959,
    "merced": 0.246938,
    "disco": 0.218514,
    "responder": 0.202375,
    "decir": 0.185875,
    "si": 0.179109,
    "madre": 0.174555,
    "departamento": 0.155260,
    "indio": 0.146644,
    "gratin": 0.145049,
    "hall": 0.145049,
    "music": 0.145049,
    "tíaú": 0.140762,
    "caballo": 0.140434,
    "cosa": 0.139339,
    "querío": 0.132032,
    "jesuita": 0.125143,
    "andante": 0.122461,
    "haber": 0.117470,
    "nombre": 0.109780,
    "historia": 0.107609,
    "hombre": 0.104552,
    "nativo": 0.097764,
    "religioso": 0.096795,
    "aventura": 0.096324,
    "personal": 0.096142,
    "expedición": 0.095940,
    "bueno": 0.094827,
    "ver": 0.094610,
    "asno": 0.093899,
    "guaraníe": 0.093857,
    "atreviste": 0.092886,
    "insumiso": 0.092886,
    "venta": 0.091239,
    "moto": 0.090189,
    "industria": 0.090170,
    "sortijo": 0.089573,
    "belle": 0.087030,
    "époque": 0.087030,
    "ventero": 0.086345,
    "dama": 0.085750,
    "aunque": 0.085573,
    "querer": 0.083388,
    "cuyo": 0.082547,
    "sistema": 0.081583,
    "poner": 0.081580,
    "pronto": 0.081035,
    "social": 0.080370,
    "artístico": 0.080299,
    "propio": 0.078760,
    "europeo": 0.078614,
    "arte": 0.077693,
    "embarcación": 0.077443,
    "polinesio": 0.077443,
    "tronco": 0.077443,
    "caballería": 0.076237,
    "así": 0.075571,
    "arqueólogo": 0.075517,
    "civilización": 0.074553,
    "triste": 0.074523,
    "ir": -0.180310,
    "mosén": -0.144567,
    "cuadro": -0.138068,
    "chico": -0.137282,
    "español": -0.135137,
    "mary": -0.125490,
    "luis": -0.124486,
    "camping": -0.118361,
    "marinero": -0.118165,
    "bar": -0.112789,
    "museo": -0.107938,
    "plaza": -0.106300,
    "agencia": -0.105839,
    "nuevo": -0.104224,
    "bolsa": -0.101858,
    "coger": -0.098086,
    "socio": -0.098063,
    "calle": -0.094810,
    "sinagoga": -0.094165,
    "ventana": -0.093580,
    "siempre": -0.092933,
    "tener": -0.092436,
    "calor": -0.090040,
    "mañana": -0.090015,
    "verónicamente": -0.088294,
    "llave": -0.087497,
    "poder": -0.087330,
    "pez": -0.086576,
    "sentar": -0.086053,
    "dentro": -0.083506,
    "cerca": -0.083421,
    "empezar": -0.083120,
    "apartamento": -0.082815,
    "sol": -0.082254,
    "piso": -0.082060,
    "lunes": -0.081911,
    "perro": -0.081302,
    "pensar": -0.080969,
    "marido": -0.080097,
    "nervioso": -0.079642,
    "puerta": -0.079533,
    "ciego": -0.078939,
    "fiesta": -0.078759,
    "rey": -0.077700,
    "niño": -0.077560,
    "billete": -0.077504,
    "dos": -0.077380,
    "alcalde": -0.076561,
    "arca": -0.076392,
    "río": -0.075182,
    "ojo": -0.074623,
    "café": -0.074421,
    "hora": -0.072329,
    "casa": -0.071301,
    "allí": -0.070518,
    "papel": -0.070514,
    "escalera": -0.070132,
    "edificio": -0.069891,
    "nadie": -0.069187,
    "despacho": -0.068544,
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
    "príncipe": 0.289273,
    "bosque": 0.248431,
    "lobo": 0.241159,
    "niña": 0.204715,
    "bruja": 0.199564,
    "soldadito": 0.194956,
    "clancanieves": 0.169126,
    "fábrica": 0.163121,
    "princesa": 0.150277,
    "ogro": 0.149350,
    "volar": 0.147334,
    "corneja": 0.142711,
    "día": 0.142135,
    "ricito": 0.125995,
    "madrastra": 0.124722,
    "palacio": 0.121435,
    "balleno": 0.120737,
    "niño": 0.119897,
    "barco": 0.116776,
    "mar": 0.111316,
    "nuevo": 0.109834,
    "castillo": 0.107638,
    "oveja": 0.099674,
    "cabrito": 0.097971,
    "cisne": 0.095720,
    "hada": 0.095409,
    "invento": 0.095286,
    "enanito": 0.093464,
    "abuela": 0.091605,
    "cada": 0.091389,
    "cazador": 0.091321,
    "madera": 0.089832,
    "pasaron": 0.088742,
    "aprender": 0.088717,
    "cole": 0.086081,
    "coronavir": 0.086081,
    "espantapájaro": 0.085925,
    "carabás": 0.085879,
    "cuento": 0.084360,
    "ayudar": 0.081819,
    "buscar": 0.081172,
    "querido": 0.079828,
    "comer": 0.078646,
    "rey": 0.078052,
    "convertir": 0.077460,
    "animal": 0.076713,
    "comenzar": 0.076631,
    "entrenar": 0.076513,
    "carlito": 0.073657,
    "nin": 0.072933,
    "trenza": 0.071742,
    "así": 0.071445,
    "taller": 0.068677,
    "osa": 0.068265,
    "polar": 0.068265,
    "mágico": 0.067846,
    "pezun": 0.067826,
    "poder": 0.067210,
    "marqués": 0.066920,
    "dinosaurios": 0.066406,
    "ser": -0.790424,
    "haber": -0.676001,
    "decir": -0.542575,
    "tener": -0.247654,
    "estudiante": -0.239071,
    "cohete": -0.234698,
    "ruiseñor": -0.232005,
    "alguno": -0.219989,
    "molinero": -0.209586,
    "rosa": -0.204295,
    "mano": -0.189713,
    "padre": -0.177632,
    "vez": -0.165217,
    "tiempo": -0.151815,
    "pelo": -0.148528,
    "pensar": -0.148067,
    "nasar": -0.146658,
    "saber": -0.146237,
    "madre": -0.141658,
    "rojo": -0.141549,
    "mismo": -0.140599,
    "santiago": -0.139674,
    "preguntar": -0.137515,
    "loco": -0.133949,
    "estatua": -0.133257,
    "habitación": -0.132054,
    "parecer": -0.131268,
    "sólo": -0.122361,
    "señora": -0.121020,
    "ir": -0.119668,
    "carretilla": -0.117781,
    "clase": -0.117040,
    "vigilante": -0.115970,
    "secreto": -0.115444,
    "páramo": -0.112520,
    "tal": -0.111818,
    "hombre": -0.111610,
    "san": -0.111256,
    "rostro": -0.109971,
    "embargo": -0.108824,
    "nunca": -0.108784,
    "cierto": -0.108125,
    "sentir": -0.108104,
    "hablar": -0.107209,
    "luego": -0.107175,
    "supervisor": -0.105427,
    "dar": -0.104294,
    "creer": -0.102118,
    "aquel": -0.101241,
    "año": -0.101028,
    "después": -0.100764,
    "noche": -0.099952,
    "dos": -0.099521,
    "pues": -0.097733,
    "morir": -0.096900,
    "replicar": -0.096538,
    "sangre": -0.096203,
    "bien": -0.095713,
    "muñeco": -0.094236,
    "último": -0.093720,
}
bow_litera_1_pos = {k: v for k, v in bow_litera_1_all.items() if v > 0}
bow_litera_1_neg = {k: v for k, v in bow_litera_1_all.items() if v < 0}
bow_litera_2_all = {
    "ser": 0.360926,
    "cohete": 0.351091,
    "molinero": 0.343267,
    "rosa": 0.329774,
    "ruiseñor": 0.308325,
    "decir": 0.289518,
    "rojo": 0.220754,
    "tener": 0.214961,
    "loco": 0.205483,
    "mano": 0.200975,
    "pelo": 0.200412,
    "estatua": 0.199097,
    "padre": 0.178658,
    "muñeco": 0.177464,
    "haber": 0.171910,
    "carretilla": 0.168875,
    "nieve": 0.139239,
    "gritar": 0.137078,
    "espino": 0.129563,
    "rubí": 0.124918,
    "replicar": 0.120914,
    "sristino": 0.116409,
    "cabeza": 0.115826,
    "rosal": 0.115521,
    "nunca": 0.109573,
    "realmente": 0.109000,
    "pardillo": 0.106658,
    "feliz": 0.105424,
    "ojo": 0.104040,
    "corazón": 0.101947,
    "voló": 0.101132,
    "exclamar": 0.097707,
    "verdad": 0.096618,
    "nota": 0.095951,
    "río": 0.094997,
    "señora": 0.091892,
    "caer": 0.091721,
    "gran": 0.090944,
    "estudiante": 0.090810,
    "rata": 0.090123,
    "pálido": 0.090093,
    "jardín": 0.089901,
    "saber": 0.089554,
    "candela": 0.087773,
    "ciudad": 0.085330,
    "bien": 0.085163,
    "llevar": 0.084143,
    "locura": 0.083702,
    "tarta": 0.083389,
    "rostro": 0.082392,
    "puse": 0.081958,
    "amistad": 0.081410,
    "secreto": 0.078464,
    "gloria": 0.077830,
    "cerda": 0.077606,
    "cara": 0.076899,
    "preguntar": 0.076787,
    "poner": 0.076407,
    "negro": 0.076112,
    "sangre": 0.074754,
    "bosque": -0.116535,
    "día": -0.114170,
    "lobo": -0.113460,
    "príncipe": -0.107008,
    "soldadito": -0.100100,
    "bruja": -0.098979,
    "encontrar": -0.090731,
    "clancanieves": -0.089721,
    "fábrica": -0.084557,
    "mujer": -0.083855,
    "ogro": -0.078135,
    "corneja": -0.077118,
    "nasar": -0.075357,
    "vigilante": -0.074243,
    "santiago": -0.071769,
    "pueblo": -0.071020,
    "comenzar": -0.070867,
    "cierto": -0.069911,
    "madrastra": -0.068226,
    "barco": -0.068220,
    "supervisor": -0.067494,
    "ricito": -0.064433,
    "don": -0.063606,
    "balleno": -0.061764,
    "mar": -0.061590,
    "niña": -0.060174,
    "volar": -0.059388,
    "páramo": -0.059380,
    "buscar": -0.058348,
    "poder": -0.057925,
    "hada": -0.057832,
    "mayor": -0.057500,
    "portero": -0.055276,
    "casa": -0.052323,
    "decidir": -0.051949,
    "princesa": -0.051637,
    "escena": -0.051395,
    "oveja": -0.051230,
    "castillo": -0.050753,
    "entrar": -0.050392,
    "cabrito": -0.049703,
    "enanito": -0.049583,
    "aprender": -0.049566,
    "hora": -0.048672,
    "cazador": -0.048341,
    "allí": -0.047332,
    "posibilidad": -0.047008,
    "marqués": -0.046771,
    "tierra": -0.046682,
    "abuela": -0.046400,
    "invento": -0.046383,
    "cada": -0.046088,
    "carabás": -0.045910,
    "pasaron": -0.045826,
    "fué": -0.045777,
    "muchacha": -0.045753,
    "hombre": -0.045353,
    "comer": -0.044982,
    "salir": -0.044891,
    "ivanovna": -0.044142,
}
bow_litera_2_pos = {k: v for k, v in bow_litera_2_all.items() if v > 0}
bow_litera_2_neg = {k: v for k, v in bow_litera_2_all.items() if v < 0}
bow_litera_3_all = {
    "haber": 0.504091,
    "ser": 0.429498,
    "decir": 0.253057,
    "nasar": 0.222015,
    "santiago": 0.211443,
    "vigilante": 0.190213,
    "cierto": 0.178036,
    "tiempo": 0.174873,
    "supervisor": 0.172921,
    "páramo": 0.171900,
    "hombre": 0.156963,
    "san": 0.155352,
    "tal": 0.154383,
    "don": 0.153843,
    "estudiante": 0.148262,
    "alguno": 0.147750,
    "portero": 0.145080,
    "román": 0.126866,
    "sólo": 0.125149,
    "fué": 0.122744,
    "bayardo": 0.121580,
    "vez": 0.121271,
    "posibilidad": 0.120785,
    "madre": 0.117883,
    "sino": 0.117689,
    "clase": 0.115511,
    "drama": 0.113771,
    "tomar": 0.113063,
    "patio": 0.111243,
    "obispo": 0.111008,
    "pueblo": 0.110649,
    "ivanovna": 0.110453,
    "después": 0.106974,
    "aquel": 0.105979,
    "si": 0.104524,
    "embargo": 0.103840,
    "oficina": 0.102786,
    "escena": 0.101735,
    "familia": 0.101027,
    "dos": 0.099000,
    "año": 0.097423,
    "estudiar": 0.096714,
    "parecer": 0.096320,
    "modo": 0.096147,
    "enfermera": 0.095585,
    "matar": 0.095176,
    "sentir": 0.094566,
    "mes": 0.093453,
    "acto": 0.093223,
    "mismo": 0.092667,
    "calle": 0.092346,
    "pensar": 0.089986,
    "director": 0.089747,
    "morir": 0.088373,
    "catalina": 0.088362,
    "querer": 0.086100,
    "cuatro": 0.085894,
    "último": 0.085851,
    "enemigo": 0.085618,
    "gente": 0.084107,
    "príncipe": -0.182264,
    "niña": -0.144541,
    "molinero": -0.133680,
    "bosque": -0.131896,
    "lobo": -0.127699,
    "rosa": -0.125479,
    "cohete": -0.116393,
    "niño": -0.114130,
    "bruja": -0.100585,
    "princesa": -0.098640,
    "soldadito": -0.094857,
    "rey": -0.093528,
    "nieve": -0.091145,
    "volar": -0.087946,
    "muñeco": -0.083228,
    "palacio": -0.080715,
    "gritar": -0.079707,
    "clancanieves": -0.079405,
    "rojo": -0.079205,
    "fábrica": -0.078564,
    "ruiseñor": -0.076320,
    "jardín": -0.074508,
    "nuevo": -0.073057,
    "oro": -0.072296,
    "loco": -0.071533,
    "ogro": -0.071214,
    "bello": -0.070866,
    "cabeza": -0.070639,
    "feliz": -0.067804,
    "estatua": -0.065841,
    "corneja": -0.065593,
    "ricito": -0.061562,
    "balleno": -0.058973,
    "exclamar": -0.058137,
    "voló": -0.057880,
    "corazón": -0.056914,
    "castillo": -0.056885,
    "madrastra": -0.056496,
    "madera": -0.056470,
    "pequeño": -0.055979,
    "cisne": -0.055385,
    "cuento": -0.054559,
    "baile": -0.052539,
    "plomo": -0.052099,
    "pelo": -0.051885,
    "empezar": -0.051760,
    "carretilla": -0.051094,
    "compartir": -0.050609,
    "mar": -0.049726,
    "espino": -0.049537,
    "descubrir": -0.049322,
    "invento": -0.048903,
    "convertir": -0.048696,
    "saco": -0.048648,
    "barco": -0.048556,
    "oveja": -0.048443,
    "cabrito": -0.048268,
    "rosal": -0.047638,
    "miedo": -0.047438,
    "ayudar": -0.047178,
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


# Command line join background image with generated overlay words
# magick convert -page 2800x2800+0+0 bg.jpg -page +0+0 bi1.png -page +0+0 bi2.png -page +0+0 bi3.png -page +0+0 bo1.png -page +0+0 bo2.png -page +0+0 bo3.png -page +0+0 ti1.png -page +0+0 ti2.png -page +0+0 ti3.png -page +0+0 to1.png -page +0+0 to2.png -page +0+0 to3.png -background white -flatten output.jpg

print("Done!")
