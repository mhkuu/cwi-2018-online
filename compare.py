from models.baseline import Baseline
from models.length import Length
from models.length_and_freq import LengthFreq
from models.logreg import LogReg
from models.ngram_missing import NgramMissing
from models.ngram_prob import NgramProb
from models.dummy import Dummy
from utils.dataset import load_data
from utils.scorer import train_and_report


def process(language):
    data = load_data(language, verbose=True)

    scores = dict()

    title = 'Baseline (normalized length)'
    baseline = Baseline(language)
    macro_f1, _ = train_and_report(baseline, data)
    scores[title] = macro_f1

    print('== Length ==')
    for n in range(3, 15):
        title = 'Length <= {}'.format(n)
        model = Length(n)
        macro_f1, _ = train_and_report(model, data)
        scores[title] = macro_f1

    print('== Length + Frequency ==')
    for n in range(5, 15):
        for m in range(1, 4):
            title = 'Length <= {} + in top {}000 of frequency list'.format(n, m * 5)
            model = LengthFreq(language, n, m * 5000)
            macro_f1, _ = train_and_report(model, data)
            scores[title] = macro_f1

    print('== N-grams ==')
    for n in range(0, 10):
        title = 'N-grams, allowed missing: {}'.format(n)
        model = NgramMissing(language, n)
        macro_f1, _ = train_and_report(model, data)
        scores[title] = macro_f1
    for i in range(1, 20):
        for n in range(2, 4):
            cut_off = 10 ** -i
            title = '{}-grams, probability cut-off: {}'.format(n, cut_off)
            model = NgramProb(language, n, cut_off)
            macro_f1, _ = train_and_report(model, data)
            scores[title] = macro_f1

    print('== All-in-one ==')
    title = 'Logistic regression'
    model = LogReg(language)
    macro_f1, _ = train_and_report(model, data)
    # print(model.model.coef_)
    scores[title] = macro_f1

    title = 'Dummy'
    dummy = Dummy()
    macro_f1, _ = train_and_report(dummy, data)
    scores[title] = macro_f1

    high_scores = sorted(scores, key=scores.get, reverse=True)[:10]
    print('High scores:')
    for n, high_score in enumerate(high_scores, start=1):
        print('{}. {}, score: {:.3f}'.format(n, high_score, scores[high_score]))


if __name__ == '__main__':
    print('= English =')
    process('english')
    print('= Spanish =')
    process('spanish')
    print('= German =')
    process('german')
    print('= French =')
    process('french')


