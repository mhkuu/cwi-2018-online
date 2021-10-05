from nltk import ngrams

from .abstract import AbstractModel


class Ngram(AbstractModel):
    def __init__(self, language, max_unknown_grams):
        self.max_unknown_grams = max_unknown_grams

        self.ngram_freq = dict()
        with open('datasets/{}/freq-ngrams.txt'.format(language), 'r') as f:
            for n, line in enumerate(f):
                if n == 0:
                    continue
                columns = line.strip('\n').split('\t')
                word = columns[0]
                freq = columns[1]
                self.ngram_freq[word] = int(freq)

    def train(self, train_set):
        pass

    def test(self, test_set):
        result = []
        for sent in test_set:
            result.append(self.calc(sent['target_word']))

        return result

    def calc(self, word):
        word = word.lower()
        unknown = 0
        for ngram in ngrams(word, 2):
            target = ngram[0] + ngram[1]
            if target not in self.ngram_freq.keys():
                unknown += 1

        # if unknown > self.max_unknown_grams:
        #    print(word, unknown)

        return str(int(unknown > self.max_unknown_grams))
