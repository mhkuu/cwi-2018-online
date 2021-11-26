import math

from nltk.lm.preprocessing import pad_both_ends
from nltk.util import ngrams

from utils.ngram import retrieve_ngram_freq


class NgramCalculator(object):
    def __init__(self, language, ngram_size=2):
        self.ngram_freq = retrieve_ngram_freq(language)
        self.ngram_size = ngram_size

    def calc_word_prob(self, word):
        prob = 1

        grams = ngrams(list(pad_both_ends(word.lower(), n=self.ngram_size)), self.ngram_size)
        for gram in grams:
            gram = list(gram)
            prob += self.calc_char_prob(''.join(gram[:-1]), ''.join(gram))
        return math.exp(prob)

    def calc_char_prob(self, prev, curr):
        vocabulary_size = len(self.ngram_freq.keys())
        # We use Laplace-smoothing here -- which is not the best solution!
        freq_curr = self.ngram_freq.get(curr, 1)
        freq_prev = self.ngram_freq.get(prev, vocabulary_size) if prev else vocabulary_size
        return math.log(freq_curr / freq_prev)
