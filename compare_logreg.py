import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from models.logreg import LogReg
from utils.dataset import load_data
from utils.scorer import train_and_report


def process_logreg(data, verbose=False):
    scores = dict()

    if verbose:
        print('== Logistic regression ==')

    model = LogReg(language)
    macro_f1, _ = train_and_report(model, data)
    scores[0] = macro_f1

    if verbose:
        high_scores = sorted(scores, key=scores.get, reverse=True)[:3]
        print('High scores:')
        for n, high_score in enumerate(high_scores, start=1):
            print('{}. {}, score: {:.3f}'.format(n, 'Logistic regression', scores[high_score]))

    return scores


if __name__ == '__main__':
    languages = ['english', 'spanish', 'german']
    all_scores = []

    # Add previous best scores
    for n, pb in enumerate([0.732, 0.667, 0.729]):
        all_scores.append({
            'language': languages[n],
            'type': 'Previous best',
            'F1-score': pb,
        })

    for language in languages:
        print('= {} ='.format(language.capitalize()))

        data = load_data(language, verbose=True)

        scores = process_logreg(data, verbose=True)
        for k, v in scores.items():
            all_scores.append({
                'language': language,
                'type': 'Logistic regression',
                'F1-score': v,
            })

    df = pd.DataFrame(all_scores)
    sns.barplot(x='language', y='F1-score', data=df, hue='type')
    plt.savefig('out/logreg.png')
