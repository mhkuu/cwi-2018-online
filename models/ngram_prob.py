import math

from utils.ngram import retrieve_ngram_freq
from .abstract import AbstractModel


class NgramProb(AbstractModel):
    def __init__(self, language, cut_off):
        self.cut_off = cut_off
        self.ngram_freq = retrieve_ngram_freq(language)

    def train(self, train_set):
        pass

    def test(self, test_set):
        result = []
        for sent in test_set:
            result.append(self.calc(sent['target_word']))

        return result

    def calc(self, word):
        # Consider upper-case words to be names and hence non-complex
        if word[0].isupper():
            return '0'

        prob = 1
        prev = '<s>'
        for character in word.lower():
            prob += self.get_ngram_prob(character, prev)
            prev = character
        prob += self.get_ngram_prob(prev, '</s>')

        # if math.exp(prob) > self.cut_off:
        #     print(word)

        return str(int(math.exp(prob) < self.cut_off))

    def get_ngram_prob(self, curr, prev):
        # We use Laplace-smoothing here -- which is not the best solution!
        prob_curr = self.ngram_freq.get(prev + curr, 1)
        prob_prev = self.ngram_freq.get(prev, len(self.ngram_freq.keys()))
        return math.log(prob_curr / prob_prev)
