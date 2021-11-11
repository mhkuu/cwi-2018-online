import math

from utils.ngram import retrieve_ngram_freq


class NgramCalculator(object):
    def __init__(self, language):
        self.ngram_freq = retrieve_ngram_freq(language)

    def calc_word_prob(self, word):
        prob = 1
        prev = '<s>'
        for character in word.lower():
            prob += self.calc_char_prob(character, prev)
            prev = character
        prob += self.calc_char_prob(prev, '</s>')
        return math.exp(prob)

    def calc_char_prob(self, curr, prev):
        # We use Laplace-smoothing here -- which is not the best solution!
        prob_curr = self.ngram_freq.get(prev + curr, 1)
        prob_prev = self.ngram_freq.get(prev, len(self.ngram_freq.keys()))
        return math.log(prob_curr / prob_prev)
