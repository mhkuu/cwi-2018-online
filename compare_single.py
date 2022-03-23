from models.baseline import Baseline
from models.length import Length
from models.length_and_freq import LengthFreq
from models.logreg import LogReg
from models.ngram_missing import NgramMissing
from models.ngram_prob import NgramProb
from models.dummy import Dummy
from utils.dataset import Dataset
from utils.scorer import report_score, train_and_report, print_results


def process(language):
    data = Dataset(language)

    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))

    scores = dict()

    title = 'Baseline (normalized length)'
    baseline = Baseline(language)
    macro_f1, _ = train_and_report(baseline, data)
    scores[title] = macro_f1

    print('== All-in-one ==')
    title = 'Logistic regression'
    model = LogReg(language)
    macro_f1, _ = train_and_report(model, data)
    print(model.model.coef_)
    scores[title] = macro_f1

    high_scores = sorted(scores, key=scores.get, reverse=True)[:3]
    print('High scores:')
    for n, high_score in enumerate(high_scores, start=1):
        print('{}. {}, score: {:.3f}'.format(n, high_score, scores[high_score]))

    # model = NgramProb(language, 3, 10 ** -6)
    # macro_f1, predictions = train_and_report(model, data, detailed=True)
    # print_results(data, predictions)


if __name__ == '__main__':
    print('= English =')
    process('english')
    print('= Spanish =')
    process('spanish')
    print('= German =')
    process('german')


