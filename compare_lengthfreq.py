import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from compare_length import process as process_length
from models.length_and_freq import LengthFreq
from utils.dataset import load_data
from utils.scorer import train_and_report


def process_length_freq(language, data, cut_off, verbose=False):
    scores = dict()

    if verbose:
        print('== Length + Frequency ==')

    for n in range(3, 15):
        model = LengthFreq(language, n, cut_off)
        macro_f1, _ = train_and_report(model, data)
        scores[n] = macro_f1

    if verbose:
        high_scores = sorted(scores, key=scores.get, reverse=True)[:3]
        print('High scores:')
        for n, high_score in enumerate(high_scores, start=1):
            print('{}. {}, score: {:.3f}'.format(n, 'Length <= {} + in top {} of frequency list'.format(high_score, cut_off), scores[high_score]))

    return scores


if __name__ == '__main__':
    languages = ['english', 'german', 'spanish', 'french']
    all_scores = []
    for language in languages:
        print('= {} ='.format(language.capitalize()))

        data = load_data(language, verbose=True)
        scores = process_length(data)
        for k, v in scores.items():
            all_scores.append({
                'language': language,
                'model': 'length',
                'length': k,
                'F1-score': v,
            })

        for cut_off in [5000, 10000]:
            scores = process_length_freq(language, data, cut_off=cut_off, verbose=True)
            for k, v in scores.items():
                all_scores.append({
                    'language': language,
                    'model': 'length + top-{}'.format(cut_off),
                    'length': k,
                    'F1-score': v,
                })

    df = pd.DataFrame(all_scores)
    sns.lineplot(x='length', y='F1-score', data=df, hue='language', style='model')
    plt.savefig('out/frequency.svg', format='svg')
