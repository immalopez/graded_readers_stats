import re

import pandas as pd
from nltk.corpus import cess_esp


def download_native_corpus(args):
    words_esp = cess_esp.words()
    text = ' '.join([word.replace('_', ' ')
                     for word in words_esp])

    # Sanitize metadata
    text = re.sub(r'-[Ff]pa-', '(', text)
    text = re.sub(r'-[Ff]pt-', ')', text)
    text = re.sub(r'\*0\*', '', text)
    text = re.sub(r'-[Ff]e-', '', text)
    text = re.sub(r' , ', ', ', text)
    text = re.sub(r' \. ', '. ', text)
    text = re.sub(r' - - ', ' ', text)
    text = re.sub(r' \? ', '? ', text)
    text = re.sub(r' ¿ ', ' ¿', text)
    text = re.sub(r' ! ', '! ', text)
    text = re.sub(r' \( ', ' (', text)
    text = re.sub(r' \) ', ') ', text)
    text = re.sub(r' \).', ').', text)
    text = re.sub(r' \),', '),', text)

    with open('../../data/native/text.txt', 'w') as f:
        f.write(text)

    df = pd.DataFrame({
        'Level': ['Native'],
        'Text file': ['data/native/text.txt']
    })
    df.to_csv('data/native.csv', index=False, sep=';')
    print('Download finished successfully!')
