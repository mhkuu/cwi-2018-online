from .freq import FrequencyCalculator


class LengthFreq(FrequencyCalculator):
    def __init__(self, language, length_cutoff):
        super().__init__(language)
        self.length_cutoff = length_cutoff

    def train(self, train_set):
        pass

    def test(self, test_set):
        result = []
        for sent in test_set:
            result.append(self.calc(sent['target_word']))

        return result

    def calc(self, word):
        stemmed = self.stemmer.stem(word, to_lowercase=True)
        result = len(word) >= self.length_cutoff and stemmed not in self.frequency.keys()
        return str(int(result))
