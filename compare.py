from models.baseline import Baseline
from models.length import Length
from models.length_and_freq import LengthFreq
from models.ngram import Ngram
from models.dummy import Dummy
from utils.dataset import Dataset
from utils.scorer import report_score


def process(language):
    data = Dataset(language)

    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))

    print('Baseline (normalized length)')
    baseline = Baseline(language)
    train_and_report(baseline, data)

    print('== Length ==')
    for n in range(3, 15):
        print('Length <= {}'.format(n))
        model = Length(n)
        predictions = train_and_report(model, data)

    if language == 'english':  # no frequency data for Spanish yet
        print('== Length + Frequency ==')
        for n in range(3, 15):
            print('Length <= {} + frequency'.format(n))
            model = LengthFreq(n)
            predictions = train_and_report(model, data)

    print('== N-grams ==')
    for n in range(0, 10):
        print('N-grams, allowed missing: {}'.format(n))
        model = Ngram(language, n)
        predictions = train_and_report(model, data)

    # print_results(data, predictions)

    print('Dummy')
    dummy = Dummy()
    train_and_report(dummy, data)


def train_and_report(model, data, detailed=False):
    model.train(data.trainset)
    predictions = model.test(data.testset)
    gold_labels = [sent['gold_label'] for sent in data.testset]
    report_score(gold_labels, predictions, detailed=detailed)
    return predictions


def print_results(data, predictions):
    for n, sent in enumerate(data.testset):
        if sent['gold_label'] == '0' and predictions[n] == '1':
            print(sent['target_word'], sent['gold_label'], predictions[n])
        if n == 1000:
            break


if __name__ == '__main__':
    process('english')
    # process('spanish')


