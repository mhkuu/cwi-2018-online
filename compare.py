from models.baseline import Baseline
from models.length import Length
from models.length_and_freq import LengthFreq
from models.logreg import LogReg
from models.ngram_missing import NgramMissing
from models.ngram_prob import NgramProb
from models.dummy import Dummy
from utils.dataset import Dataset
from utils.scorer import report_score


def process(language):
    data = Dataset(language)

    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))

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

    if language in ['english', 'spanish', 'german']:
        print('== Length + Frequency ==')
        for n in range(3, 15):
            title = 'Length <= {} + frequency'.format(n)
            model = LengthFreq(language, n)
            macro_f1, _ = train_and_report(model, data)
            scores[title] = macro_f1

    print('== N-grams ==')
    for n in range(0, 10):
        title = 'N-grams, allowed missing: {}'.format(n)
        model = NgramMissing(language, n)
        macro_f1, _ = train_and_report(model, data)
        scores[title] = macro_f1
    for n in range(10, 20):
        cut_off = 10 ** -n
        title = 'N-grams, probability cut-off: {}'.format(cut_off)
        model = NgramProb(language, cut_off)
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


def train_and_report(model, data, detailed=False):
    model.train(data.trainset)
    predictions = model.test(data.testset)
    gold_labels = [sent['gold_label'] for sent in data.testset]
    macro_f1 = report_score(gold_labels, predictions, detailed=detailed)
    return macro_f1, predictions


def print_results(data, predictions):
    for n, sent in enumerate(data.testset):
        if sent['gold_label'] != predictions[n]:
            print(sent['target_word'], sent['gold_label'], predictions[n])
        if n == 1000:
            break


if __name__ == '__main__':
    print('= English =')
    process('english')
    print('= Spanish =')
    process('spanish')
    print('= German =')
    process('german')
    # print('= French =')
    # process('french')


