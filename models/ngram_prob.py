from .abstract import AbstractModel
from calculators.ngram import NgramCalculator


class NgramProb(AbstractModel):
    def __init__(self, language, ngram_size, cut_off):
        self.language = language
        self.ngram_size = ngram_size
        self.cut_off = cut_off

        self.ngram_calc = NgramCalculator(language, ngram_size)

    def train(self, train_set):
        # No training required
        pass

    def test(self, test_set):
        result = []
        for sent in test_set:
            result.append(self.calc(sent['target_word']))

        return result

    def calc(self, word):
        # Consider upper-case words in languages other than German to be names and hence non-complex
        if self.language != 'german' and word[0].isupper():
            return '0'

        prob = self.ngram_calc.calc_word_prob(word)

        return str(int(prob < self.cut_off))
