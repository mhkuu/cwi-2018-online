import unittest

from calculators.ngram import NgramCalculator


class TestNgramCalculator(unittest.TestCase):

    def test_unigram(self):
        self.ngram_calc = NgramCalculator('english', 1)

    def test_bigram(self):
        self.ngram_calc = NgramCalculator('english', 2)
        self.assertEqual(4.5655253985493955e-05, self.ngram_calc.calc_word_prob('test'))

    def test_trigram(self):
        self.ngram_calc = NgramCalculator('english', 3)
        self.assertEqual(3.6777911019944145e-05, self.ngram_calc.calc_word_prob('test'))
