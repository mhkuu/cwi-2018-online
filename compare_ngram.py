import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from models.ngram_prob import NgramProb
from utils.dataset import load_data
from utils.scorer import train_and_report


def process_ngram(data, ngram_size, verbose=False):
    scores = dict()

    if verbose:
        print('== n-gram probability ==')

    for i in range(1, 20):
        cut_off = 10 ** -i
        model = NgramProb(language, ngram_size=ngram_size, cut_off=cut_off)
        macro_f1, _ = train_and_report(model, data)
        scores[i] = macro_f1

    if verbose:
        high_scores = sorted(scores, key=scores.get, reverse=True)[:3]
        print('High scores:')
        for n, high_score in enumerate(high_scores, start=1):
            print('{}. {}, score: {:.3f}'.format(n, '{}-grams, probability cut-off: {}'.format(ngram_size, high_score), scores[high_score]))

    return scores


if __name__ == '__main__':
    languages = ['english', 'spanish', 'german', 'french']
    all_scores = []
    for language in languages:
        print('= {} ='.format(language.capitalize()))

        data = load_data(language, verbose=True)

        for ngram_size in range(2, 4):
            scores = process_ngram(data, ngram_size=ngram_size, verbose=True)
            for k, v in scores.items():
                all_scores.append({
                    'language': language,
                    'model': '{}-grams'.format(ngram_size),
                    'cut-off': k,
                    'F1-score': v,
                })

    df = pd.DataFrame(all_scores)
    sns.lineplot(x='cut-off', y='F1-score', data=df, hue='language', style='model')
    plt.savefig('out/ngram-prob.png')
