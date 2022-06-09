import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from compare_lengthfreq import process_length_freq
from compare_ngram import process_ngram
from models.logreg import LogReg
from utils.dataset import load_data
from utils.scorer import train_and_report


def process_logreg(language, data, verbose=False):
    scores = dict()

    if verbose:
        print('== Logistic regression ==')

    if language == 'french':
        print('No test data for French')
        scores[0] = 0
        return scores

    model = LogReg(language)
    macro_f1, _ = train_and_report(model, data)
    scores[0] = macro_f1
    if verbose:
        print('Model score:', macro_f1)
        print('Model coefficients:', model.model.coef_)

    return scores


if __name__ == '__main__':
    languages = ['english', 'german', 'spanish', 'french']
    all_scores = []

    for language in languages:
        print('= {} ='.format(language.capitalize()))

        data = load_data(language, verbose=True)

        cut_off = 5000
        scores = process_length_freq(language, data, cut_off=cut_off, verbose=True)
        all_scores.append({
            'language': language,
            'model': 'length + top-{}'.format(cut_off),
            'F1-score': max(scores.values()),
        })

        ngram_size = 3
        scores = process_ngram(language, data, ngram_size=ngram_size, verbose=True)
        all_scores.append({
            'language': language,
            'model': 'character {}-gram prob.'.format(ngram_size),
            'F1-score': max(scores.values()),
        })

        scores = process_logreg(language, data, verbose=True)
        for k, v in scores.items():
            all_scores.append({
                'language': language,
                'model': 'logistic regression',
                'F1-score': v,
            })

    # Add SOTA
    for n, pb in enumerate([0.8736, 0.7451, 0.7699, 0.7595]):
        all_scores.append({
            'language': languages[n],
            'model': 'state-of-the-art',
            'F1-score': pb,
        })

    df = pd.DataFrame(all_scores)
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='language', y='F1-score', hue='model')
    ax.set(ylim=(.6, .9))
    plt.savefig('out/comparison.svg', format='svg')
