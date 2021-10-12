from .ngram import NgramCalculator


class NgramProb(NgramCalculator):
    def __init__(self, language, cut_off):
        super().__init__(language)
        self.cut_off = cut_off

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

        prob = self.full_ngram_prob(word)

        # if math.exp(prob) > self.cut_off:
        #     print(word)

        return str(int(prob < self.cut_off))
