from nltk import ngrams

from .abstract import AbstractModel
from calculators.ngram import NgramCalculator


class NgramMissing(AbstractModel):
    def __init__(self, language, max_unknown_grams):
        self.language = language
        self.max_unknown_grams = max_unknown_grams

        self.ngram_calc = NgramCalculator(language)

    def train(self, train_set):
        # No training required
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
            if target not in self.ngram_calc.ngram_freq.keys():
                unknown += 1

        # if unknown > self.max_unknown_grams:
        #    print(word, unknown)

        return str(int(unknown > self.max_unknown_grams))
