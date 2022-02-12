import ntpath
import os

import pandas as pd


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def merge_output(args):
    cwd = os.path.abspath('./output')
    files = os.listdir(cwd)
    output = pd.DataFrame()
    for file in files:
        if not file.endswith('.csv') or file.endswith('main.csv'):
            continue

        level, _ = path_leaf(file).split('.')
        df = pd.read_csv(f'./output/{file}')
        df = df.drop(columns=[
            'Context words',
            'Context count per word'
        ])
        # Avoid re-adding shared columns such as Lexical item, Topic, Subtopic
        if len(output) > 0:
            df = df.drop(columns=[
                'Lexical item',
                'Level',
                'Topic',
                'Subtopic',
                'Lemma',
            ])
        df = df.rename(columns={
            "Count": f"Count_{level}",
            "Total": f"Total_{level}",
            "Frequency": f"Frequency_{level}",
            "TFIDF": f"TFIDF_{level}",
            "Tree": f"Tree_{level}",

            "Context count": f"Context_count_{level}",
            "Context total": f"Context_total_{level}",
            "Context frequency": f"Context_frequency_{level}",
            "Context TFIDF": f"Context_TFIDF_{level}",
            "Context tree": f"Context_tree_{level}",
        })
        output = pd.concat([output, df], axis=1)

    output.head()
    output.to_csv(f'./output/main.csv')
