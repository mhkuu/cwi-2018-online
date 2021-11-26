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

    print('== N-grams ==')
    for n in range(0, 10):
        title = 'N-grams, allowed missing: {}'.format(n)
        model = NgramMissing(language, n)
        macro_f1, _ = train_and_report(model, data)
        scores[title] = macro_f1
        print(n, macro_f1)
    for i in range(5, 20):
        for n in range(2, 4):
            cut_off = 10 ** -i
            title = '{}-grams, probability cut-off: {}'.format(n, cut_off)
            model = NgramProb(language, n, cut_off)
            macro_f1, _ = train_and_report(model, data)
            scores[title] = macro_f1
            print(n, i, macro_f1)

    high_scores = sorted(scores, key=scores.get, reverse=True)[:3]
    print('High scores:')
    for n, high_score in enumerate(high_scores, start=1):
        print('{}. {}, score: {:.3f}'.format(n, high_score, scores[high_score]))

    model = NgramProb(language, 3, 10 ** -6)
    macro_f1, predictions = train_and_report(model, data, detailed=True)
    print_results(data, predictions)


def train_and_report(model, data, detailed=False):
    model.train(data.trainset)
    predictions = model.test(data.testset)
    gold_labels = [sent['gold_label'] for sent in data.testset]
    macro_f1 = report_score(gold_labels, predictions, detailed=detailed)
    return macro_f1, predictions


def print_results(data, predictions):
    for n, sent in enumerate(data.testset):
        if len(sent['target_word']) >= 10 and sent['gold_label'] == predictions[n]:
            print(sent['target_word'], sent['gold_label'], predictions[n])
        if n == 1000:
            break


if __name__ == '__main__':
    print('= German =')
    process('english')
    # print('= Spanish =')
    # process('spanish')


