import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from models.length import Length
from utils.dataset import load_data
from utils.scorer import train_and_report


def process(data):
    scores = dict()

    print('== Length ==')
    for n in range(3, 15):
        model = Length(n)
        macro_f1, _ = train_and_report(model, data)
        scores[n] = macro_f1

    high_scores = sorted(scores, key=scores.get, reverse=True)[:3]
    print('High scores:')
    for n, high_score in enumerate(high_scores, start=1):
        print('{}. {}, score: {:.3f}'.format(n, 'Length <= {}'.format(high_score), scores[high_score]))

    return scores


if __name__ == '__main__':
    languages = ['english', 'spanish', 'german', 'french']
    all_scores = []
    for language in languages:
        print('= {} ='.format(language.capitalize()))

        data = load_data(language, verbose=True)
        scores = process(data)
        for k, v in scores.items():
            all_scores.append({
                'language': language,
                'length': k,
                'F1-score': v,
            })

    df = pd.DataFrame(all_scores)
    sns.lineplot(x='length', y='F1-score', data=df, hue='language')
    plt.savefig('out/length.png')
